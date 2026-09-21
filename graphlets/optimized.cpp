/* orbit5.cpp - per-node graphlet orbit counts for graphlets on 2..5 nodes.
 *
 * Command line and output files are identical to bruteforce_5.cpp:
 *     orbit5 <input graph> <output name>
 * writes <output name> and <output name>.gr_freq (29 graphlet totals) and
 * <output name>.ndump2 (one line per node: degree followed by 72 orbit counts).
 *
 * The enumeration visits every connected induced subgraph on 3, 4 and 5 nodes
 * exactly once, rooted at its lowest-numbered vertex, and classifies it through
 * tables that are built at start-up by canonicalising all 2^10 possible
 * adjacency masks.  Nothing about the graphlet taxonomy is hand-tabulated; the
 * only constants are the 29 canonical masks and the output column each of
 * their vertex orbits owns, which is machine-checked against the reference
 * implementation (see perf/orbits.py).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <ctime>
#include <thread>
#include <atomic>
#include <functional>

typedef long long int64;
typedef uint64_t u64;

struct Canon { unsigned mask; int g; int col[5]; int tcol; int tsize; };

static const int NCAN3 = 2;
static const Canon CAN3[2] = {
    {   3,  0, { 2, 1,-1,-1,-1},  1, 2},
    {   7,  1, { 3,-1,-1,-1,-1},  3, 3},
};
static const int NCAN4 = 6;
static const Canon CAN4[6] = {
    {   7,  3, { 7, 6,-1,-1,-1},  6, 3},
    {  13,  2, { 5,-1, 4,-1,-1},  4, 2},
    {  15,  5, {11,10,-1, 9,-1},  9, 1},
    {  30,  4, { 8,-1,-1,-1,-1},  8, 4},
    {  31,  6, {13,-1,12,-1,-1}, 12, 2},
    {  63,  7, {14,-1,-1,-1,-1}, 14, 4},
};
static const int NCAN5 = 21;
static const Canon CAN5[21] = {
    {  15, 10, {23,22,-1,-1,-1}, 22, 4},
    {  29,  9, {21,20,18,19,-1}, 18, 1},
    {  31, 13, {33,32,-1,31,-1}, 31, 2},
    {  58,  8, {16,-1,17,15,-1}, 15, 2},
    {  59, 11, {26,-1,25,24,-1}, 24, 2},
    {  62, 15, {38,36,37,-1,35}, 35, 1},
    {  63, 16, {42,41,40,-1,39}, 39, 1},
    { 126, 19, {50,-1,49,-1,-1}, 49, 3},
    { 127, 21, {55,-1,54,-1,-1}, 54, 3},
    { 185, 12, {28,30,29,-1,27}, 27, 1},
    { 187, 18, {47,48,-1,46,45}, 45, 1},
    { 191, 22, {58,57,-1,-1,56}, 56, 1},
    { 207, 17, {44,43,-1,-1,-1}, 43, 4},
    { 220, 14, {34,-1,-1,-1,-1}, 34, 5},
    { 221, 20, {53,-1,51,-1,52}, 51, 2},
    { 223, 23, {61,60,-1,59,-1}, 59, 2},
    { 254, 24, {63,-1,64,-1,62}, 62, 1},
    { 255, 25, {67,-1,66,-1,65}, 65, 1},
    { 495, 26, {69,68,-1,-1,-1}, 68, 4},
    { 511, 27, {71,-1,-1,70,-1}, 70, 2},
    {1023, 28, {72,-1,-1,-1,-1}, 72, 5},
};

static const Canon *CAN[6] = {0, 0, 0, CAN3, CAN4, CAN5};
static const int NCAN[6] = {0, 0, 0, NCAN3, NCAN4, NCAN5};

/* ---------------------------------------------------------------- taxonomy */

/* pair (i,j), i<j, occupies bit sum_{a<i}(k-1-a) + (j-i-1) of a k-node mask,
   matching the lexicographic ordering of combinations used to derive CAN*. */
static inline int pairbit(int k, int i, int j) {
    int b = 0;
    for (int a = 0; a < i; a++) b += k - 1 - a;
    return b + (j - i - 1);
}

/* classification tables: for every k-node adjacency mask, the graphlet index
   and the output column owned by each of the k positions (-1 if disconnected) */
static int8_t *cls_col[6];
static int8_t *cls_g[6];

static unsigned permute_mask(int k, unsigned mask, const int *p) {
    unsigned out = 0;
    for (int i = 0; i < k; i++)
        for (int j = i + 1; j < k; j++)
            if (mask >> pairbit(k, i, j) & 1) {
                int a = p[i], b = p[j];
                out |= 1u << pairbit(k, std::min(a, b), std::max(a, b));
            }
    return out;
}

static bool mask_connected(int k, unsigned mask) {
    int seen = 1, frontier = 1;
    while (frontier) {
        int next = 0;
        for (int i = 0; i < k; i++) if (frontier >> i & 1)
            for (int j = 0; j < k; j++)
                if (j != i && !(seen >> j & 1) &&
                    (mask >> pairbit(k, std::min(i, j), std::max(i, j)) & 1))
                    next |= 1 << j;
        seen |= next; frontier = next;
    }
    return seen == (1 << k) - 1;
}

static void build_tables() {
    int perm[5], order[5];
    for (int k = 3; k <= 5; k++) {
        int nbits = k * (k - 1) / 2, nmask = 1 << nbits;
        cls_col[k] = (int8_t *)malloc((size_t)nmask * k);
        cls_g[k] = (int8_t *)malloc(nmask);
        memset(cls_col[k], -1, (size_t)nmask * k);
        memset(cls_g[k], -1, nmask);
        for (unsigned m = 0; m < (unsigned)nmask; m++) {
            if (!mask_connected(k, m)) continue;
            /* canonical mask, and the lowest canonical slot each vertex reaches */
            unsigned best = ~0u;
            for (int i = 0; i < k; i++) order[i] = i;
            do {
                for (int i = 0; i < k; i++) perm[i] = order[i];
                unsigned c = permute_mask(k, m, perm);
                if (c < best) best = c;
            } while (std::next_permutation(order, order + k));
            int slot[5];
            for (int i = 0; i < k; i++) slot[i] = k;
            for (int i = 0; i < k; i++) order[i] = i;
            do {
                for (int i = 0; i < k; i++) perm[i] = order[i];
                if (permute_mask(k, m, perm) == best)
                    for (int i = 0; i < k; i++) slot[i] = std::min(slot[i], perm[i]);
            } while (std::next_permutation(order, order + k));
            int ci = -1;
            for (int i = 0; i < NCAN[k]; i++) if (CAN[k][i].mask == best) ci = i;
            if (ci < 0) { fprintf(stderr, "no canonical class for mask\n"); exit(1); }
            cls_g[k][m] = (int8_t)CAN[k][ci].g;
            for (int i = 0; i < k; i++)
                cls_col[k][(size_t)m * k + i] = (int8_t)CAN[k][ci].col[slot[i]];
        }
    }
}

/* ---------------------------------------------------------------- graph */

static int n;
static int64 m_undirected;
static std::vector<int> adj;      /* CSR neighbour ids, each list sorted */
static std::vector<int64> off;    /* CSR offsets, size n+1 */
static std::vector<int> deg;
static std::vector<u64> bits;     /* adjacency bitmatrix, W words per row */
/* rev[j] is the slot of the reverse of directed edge j, so a quantity that is a
   property of the undirected edge can be computed once and stored twice. */
static std::vector<int64> rev;
static size_t W;

static bool use_bits;             /* is the n^2/8-byte adjacency matrix present? */

/* Adjacency test.  The bit matrix answers in one load but costs n^2/8 bytes,
   which is 1.25 GB at n = 100k; above a size gate the sorted adjacency lists
   answer it instead, searching whichever endpoint has the shorter list. */
static inline bool linked(int a, int b) {
    if (use_bits) return bits[(size_t)a * W + (b >> 6)] >> (b & 63) & 1;
    if (deg[a] > deg[b]) { int t = a; a = b; b = t; }
    return std::binary_search(adj.begin() + off[a], adj.begin() + off[a + 1], b);
}

/* |N(a) & N(b) & N(c)|.  With the bit matrix this is a popcount over n/64
   words; without it, walking the shortest list is both smaller and, on a
   sparse graph, far cheaper - 1563 words versus about d probes at n = 100k. */
static inline int64 triple_common(int a, int b, int c);

static void read_graph(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(1); }
    buf[sz] = 0;
    fclose(f);
    const char *p = buf;
    auto num = [&p]() -> long {
        while (*p && (*p < '0' || *p > '9')) p++;
        long v = 0;
        while (*p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
        return v;
    };
    n = (int)num();
    long e = num();
    std::vector<int> ea(e), eb(e);
    long cnt = 0;
    for (long i = 0; i < e; i++) {
        int a = (int)num(), b = (int)num();
        if (a == b) continue;
        if (a >= n || b >= n) { fprintf(stderr, "node id out of range\n"); exit(1); }
        ea[cnt] = a; eb[cnt] = b; cnt++;
    }
    free(buf);
    deg.assign(n, 0);
    for (long i = 0; i < cnt; i++) { deg[ea[i]]++; deg[eb[i]]++; }
    off.assign(n + 1, 0);
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    adj.resize(off[n]);
    std::vector<int64> cur(off.begin(), off.end() - 1);
    for (long i = 0; i < cnt; i++) {
        adj[cur[ea[i]]++] = eb[i];
        adj[cur[eb[i]]++] = ea[i];
    }
    /* sort and drop parallel edges, exactly as the reference does */
    int64 w = 0;
    std::vector<int64> noff(n + 1, 0);
    for (int i = 0; i < n; i++) {
        std::sort(adj.begin() + off[i], adj.begin() + off[i + 1]);
        int64 s = off[i];
        int64 keep = w;
        for (int64 j = s; j < off[i + 1]; j++)
            if (j == s || adj[j] != adj[j - 1]) adj[w++] = adj[j];
        noff[i + 1] = w;
        deg[i] = (int)(w - keep);
    }
    off = noff;
    adj.resize(w);
    m_undirected = w / 2;
    /* Reverse slots in one linear pass: adjacency lists are sorted, so as v
       increases, the neighbours of u greater than u are queried in increasing
       order and a per-node cursor finds each in O(1). */
    rev.assign(adj.size(), 0);
    {
        std::vector<int64> cur(n);
        for (int i = 0; i < n; i++)
            cur[i] = std::upper_bound(adj.begin() + off[i], adj.begin() + off[i + 1], i)
                     - adj.begin();
        for (int v = 0; v < n; v++)
            for (int64 j = off[v]; j < off[v + 1]; j++) {
                int u = adj[j];
                if (u >= v) break;
                const int64 r = cur[u]++;
                rev[j] = r; rev[r] = j;
            }
    }
    W = (n + 63) / 64;
    /* 512 MiB gate: below it the bit matrix is worth its memory, above it the
       sorted lists are both smaller and faster on the sparse graphs that get
       that large. */
    use_bits = ((double)n * W * 8.0) <= 512.0 * 1024 * 1024;
    if (use_bits) {
        bits.assign((size_t)n * W, 0);
        for (int i = 0; i < n; i++)
            for (int64 j = off[i]; j < off[i + 1]; j++)
                bits[(size_t)i * W + (adj[j] >> 6)] |= 1ull << (adj[j] & 63);
    }
}

/* ---------------------------------------------------------------- counting */

static std::vector<int64> orbit;   /* n x 73, the merged result */

/* Per-worker state.  Roots are handed out in chunks, and each worker owns a
   private accumulator, so no increment is ever contended; the accumulators are
   summed once at the end.  The default is one worker, which uses the result
   array directly and pays neither the extra memory nor the merge. */
struct Ctx {
    int S[5];
    unsigned char A[5][5];
    int64 *acc;
    std::vector<int> scratch;
    std::vector<int> ext0;
};

static inline void bump(Ctx &c, int v, int col) { c.acc[(size_t)v * 73 + col]++; }

static void rec(Ctx &c, int root, int slen, const int *ext, int elen, int *scratch) {
    int *S = c.S;
    unsigned char (*A)[5] = c.A;
    if (slen == 4) {
        unsigned partial = A[0][1] | (unsigned)(A[0][2] << 1) | (unsigned)(A[0][3] << 2)
                         | (unsigned)(A[1][2] << 4) | (unsigned)(A[1][3] << 5)
                         | (unsigned)(A[2][3] << 7);
        int s0 = S[0], s1 = S[1], s2 = S[2], s3 = S[3];
        for (int i = 0; i < elen; i++) {
            int w = ext[i];
            const u64 *r = &bits[(size_t)w * W];
            unsigned mm = partial
                | (unsigned)((r[s0 >> 6] >> (s0 & 63) & 1) << 3)
                | (unsigned)((r[s1 >> 6] >> (s1 & 63) & 1) << 6)
                | (unsigned)((r[s2 >> 6] >> (s2 & 63) & 1) << 8)
                | (unsigned)((r[s3 >> 6] >> (s3 & 63) & 1) << 9);
            const int8_t *c5 = cls_col[5] + (size_t)mm * 5;
            bump(c, s0, c5[0]); bump(c, s1, c5[1]); bump(c, s2, c5[2]);
            bump(c, s3, c5[3]); bump(c, w, c5[4]);
        }
        return;
    }
    for (int i = 0; i < elen; i++) {
        int w = ext[i];
        for (int t = 0; t < slen; t++) {
            unsigned char x = (unsigned char)linked(w, S[t]);
            A[t][slen] = x; A[slen][t] = x;
        }
        S[slen] = w;
        int ne = 0;
        for (int j = i + 1; j < elen; j++) scratch[ne++] = ext[j];
        for (int64 j = off[w]; j < off[w + 1]; j++) {
            int u = adj[j];
            if (u <= root) continue;
            bool bad = false;
            for (int t = 0; t < slen; t++)
                if (u == S[t] || linked(u, S[t])) { bad = true; break; }
            if (!bad) scratch[ne++] = u;
        }
        rec(c, root, slen + 1, scratch, ne, scratch + ne);
    }
}

static void roots(Ctx &c, int lo, int hi) {
    for (int v = lo; v < hi; v++) {
        c.ext0.clear();
        for (int64 j = off[v]; j < off[v + 1]; j++)
            if (adj[j] > v) c.ext0.push_back(adj[j]);
        c.S[0] = v;
        if (!c.ext0.empty())
            rec(c, v, 1, c.ext0.data(), (int)c.ext0.size(), c.scratch.data());
    }
}

/* ------------------------------------------------------- algebraic engine

   Every one of the 73 output columns is computed from rooted homomorphism
   counts.  Homomorphisms factor where induced counts do not: a degree-1
   pattern vertex contributes a degree factor and disappears, and a pattern
   splits at a cut vertex into a product, so each of the 29 patterns collapses
   onto its 2-core.  What is left is 48 weighted evaluations over 16 cores -
   the triangle, C4, the diamond and K4 carrying weights of 1, d, A.d or d^2,
   and the eleven five-vertex graphs of minimum degree two carrying none.  Of
   those eleven only the 5-cycle and K(2,3) are triangle-free; every other one
   is reachable from a triangle, which is what keeps the cost down.

   perf/homplan.py proves the reduction, perf/cores5.py checks each five-vertex
   evaluation against brute force, and perf/algebra.py derives and verifies the
   rational matrix that turns homomorphism counts back into induced counts.
   perf/progtable.inc and perf/wtable.inc are the generated tables.

   The passes below are deliberately separate rather than fused; each rebuilds
   the neighbourhood structure it needs.  That costs a constant factor and
   keeps every formula readable against its derivation.

   Building with -DORBIT_NO_C5 omits the five-cycle term.  The result is wrong
   and the build exists only so that term's share of the run can be measured
   through the harness like any other build, rather than by an ad-hoc binary
   whose numbers nothing can reproduce.                                       */

/* ---- begin inlined perf/progtable.inc ---- */
/* 48 weighted core evaluations and 74 expression nodes,
   hash-consed so an identical subtree is evaluated once for the whole set.
   Generated by perf/gen_tables.py; do not edit. */
struct CoreSpec { int k; int mask; int root; int w[5]; };
static const int NCORE = 48;
static const CoreSpec CORES[] = {
    {3, 7, 0, {0,0,0,0,0}},
    {4, 30, 0, {0,0,0,0,0}},
    {3, 7, 1, {1,0,0,0,0}},
    {3, 7, 0, {1,0,0,0,0}},
    {4, 31, 2, {0,0,0,0,0}},
    {4, 31, 0, {0,0,0,0,0}},
    {4, 63, 0, {0,0,0,0,0}},
    {3, 7, 2, {1,1,0,0,0}},
    {3, 7, 0, {1,1,0,0,0}},
    {3, 7, 1, {2,0,0,0,0}},
    {3, 7, 0, {2,0,0,0,0}},
    {3, 7, 1, {3,0,0,0,0}},
    {3, 7, 0, {3,0,0,0,0}},
    {5, 220, 0, {0,0,0,0,0}},
    {4, 30, 1, {1,0,0,0,0}},
    {4, 30, 2, {1,0,0,0,0}},
    {4, 30, 0, {1,0,0,0,0}},
    {4, 31, 2, {1,0,0,0,0}},
    {4, 31, 1, {1,0,0,0,0}},
    {4, 31, 0, {1,0,0,0,0}},
    {5, 207, 1, {0,0,0,0,0}},
    {5, 207, 0, {0,0,0,0,0}},
    {4, 31, 3, {0,0,1,0,0}},
    {4, 31, 2, {0,0,1,0,0}},
    {4, 31, 0, {0,0,1,0,0}},
    {5, 126, 2, {0,0,0,0,0}},
    {5, 126, 0, {0,0,0,0,0}},
    {5, 221, 2, {0,0,0,0,0}},
    {5, 221, 4, {0,0,0,0,0}},
    {5, 221, 0, {0,0,0,0,0}},
    {5, 127, 2, {0,0,0,0,0}},
    {5, 127, 0, {0,0,0,0,0}},
    {4, 63, 1, {1,0,0,0,0}},
    {4, 63, 0, {1,0,0,0,0}},
    {5, 223, 3, {0,0,0,0,0}},
    {5, 223, 1, {0,0,0,0,0}},
    {5, 223, 0, {0,0,0,0,0}},
    {5, 254, 4, {0,0,0,0,0}},
    {5, 254, 0, {0,0,0,0,0}},
    {5, 254, 2, {0,0,0,0,0}},
    {5, 255, 4, {0,0,0,0,0}},
    {5, 255, 2, {0,0,0,0,0}},
    {5, 255, 0, {0,0,0,0,0}},
    {5, 495, 1, {0,0,0,0,0}},
    {5, 495, 0, {0,0,0,0,0}},
    {5, 511, 3, {0,0,0,0,0}},
    {5, 511, 0, {0,0,0,0,0}},
    {5, 1023, 0, {0,0,0,0,0}},
};
static const int NNODE = 74;
static const int NODES[][3] = {
    {0,0,0},
    {1,0,0},
    {1,1,0},
    {2,1,1},
    {3,0,0},
    {1,2,0},
    {2,1,2},
    {1,3,0},
    {2,1,3},
    {3,1,0},
    {1,4,0},
    {3,2,0},
    {3,3,0},
    {3,4,0},
    {3,5,0},
    {3,6,0},
    {1,5,0},
    {2,1,5},
    {2,2,2},
    {1,7,0},
    {1,6,0},
    {2,1,7},
    {2,1,6},
    {1,8,0},
    {2,1,8},
    {1,11,0},
    {3,7,0},
    {3,8,0},
    {1,10,0},
    {2,1,10},
    {3,9,0},
    {3,10,0},
    {1,12,0},
    {3,11,0},
    {3,12,0},
    {3,13,0},
    {1,9,0},
    {3,14,0},
    {3,15,0},
    {3,16,0},
    {1,14,0},
    {3,17,0},
    {3,18,0},
    {3,19,0},
    {3,20,0},
    {3,21,0},
    {1,13,0},
    {3,22,0},
    {3,23,0},
    {3,24,0},
    {3,25,0},
    {3,26,0},
    {3,27,0},
    {3,28,0},
    {3,29,0},
    {3,30,0},
    {3,31,0},
    {1,15,0},
    {3,32,0},
    {3,33,0},
    {3,34,0},
    {3,35,0},
    {3,36,0},
    {3,37,0},
    {3,38,0},
    {3,39,0},
    {3,40,0},
    {3,41,0},
    {3,42,0},
    {3,43,0},
    {3,44,0},
    {3,45,0},
    {3,46,0},
    {3,47,0},
};
static const int PROGROOT[73] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73};
/* ---- end inlined perf/progtable.inc ---- */
/* ---- begin inlined perf/wtable.inc ---- */
/* IND[c] = ( sum_y WNUM[..] * HOM[y] ) / 24; derived and verified by perf/algebra.py */
static const int WDEN = 24;
static const int WOFF[74] = {
    0, 1, 4, 7, 8, 15, 22, 31, 38, 46, 49, 53, 56, 59, 62, 63, 97, 131, 159, 194, 235, 274, 307, 336, 354, 374, 392, 412, 428, 444, 464, 480, 499, 519, 532, 550, 574, 599, 628, 653, 666, 682, 696, 707, 717, 723, 733, 744, 754, 766, 788, 805, 817, 827, 839, 848, 855, 859, 865, 869, 876, 883, 888, 897, 906, 917, 920, 924, 927, 935, 941, 944, 947, 948};
static const short WCOL[948] = {
    0, 0, 1, 3, 0, 2, 3, 3, 4, 8, 9, 10, 12, 13, 14, 5, 8, 10, 11, 12, 13, 14, 0, 1, 3, 6, 9, 10, 12, 13, 14, 0, 2, 3, 7, 11, 13, 14, 0, 1, 2, 3, 8, 12, 13, 14, 9, 12, 14, 10, 12, 13, 14, 11, 13, 14, 3, 12, 14, 3, 13, 14, 14, 15, 24, 27, 29, 34, 35, 37, 39, 40, 43, 45, 46, 48, 49, 51, 52, 53, 54, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 16, 26, 28, 29, 34, 36, 38, 41, 42, 43, 46, 47, 48, 50, 51, 52, 53, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 17, 25, 30, 34, 37, 40, 44, 48, 49, 51, 52, 53, 54, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 4, 8, 9, 10, 12, 13, 14, 18, 24, 27, 32, 36, 39, 40, 41, 43, 45, 46, 50, 51, 54, 55, 56, 57, 59, 60, 62, 63, 65, 66, 67, 68, 70, 71, 72, 4, 8, 9, 10, 12, 13, 14, 19, 24, 25, 29, 31, 35, 37, 39, 40, 43, 45, 46, 48, 49, 51, 52, 53, 54, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 5, 8, 10, 11, 12, 13, 14, 20, 26, 28, 32, 37, 40, 41, 42, 43, 47, 48, 49, 51, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 5, 8, 10, 11, 12, 13, 14, 21, 26, 30, 33, 38, 41, 42, 44, 47, 48, 50, 53, 55, 57, 58, 60, 61, 63, 64, 66, 67, 68, 69, 70, 71, 72, 0, 1, 3, 6, 9, 10, 12, 13, 14, 22, 31, 32, 39, 40, 41, 43, 54, 55, 56, 57, 59, 60, 65, 66, 67, 68, 70, 71, 72, 0, 2, 3, 7, 11, 13, 14, 23, 33, 42, 44, 55, 58, 61, 67, 69, 71, 72, 24, 39, 40, 45, 46, 51, 54, 56, 57, 59, 60, 62, 63, 65, 66, 67, 68, 70, 71, 72, 25, 40, 48, 52, 54, 57, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 26, 41, 42, 47, 48, 53, 55, 57, 58, 60, 61, 63, 64, 66, 67, 68, 69, 70, 71, 72, 27, 43, 45, 51, 56, 59, 60, 62, 63, 65, 66, 67, 68, 70, 71, 72, 28, 43, 47, 51, 58, 59, 60, 62, 63, 65, 66, 67, 68, 70, 71, 72, 29, 43, 46, 48, 52, 53, 57, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 30, 44, 48, 53, 57, 60, 61, 63, 64, 66, 67, 68, 69, 70, 71, 72, 9, 12, 14, 31, 39, 40, 43, 54, 56, 57, 59, 60, 65, 66, 67, 68, 70, 71, 72, 10, 12, 13, 14, 32, 40, 41, 43, 54, 55, 57, 59, 60, 65, 66, 67, 68, 70, 71, 72, 11, 13, 14, 33, 42, 44, 55, 58, 61, 67, 69, 71, 72, 34, 51, 52, 53, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 4, 8, 9, 10, 12, 13, 14, 35, 39, 45, 49, 52, 54, 56, 59, 62, 64, 65, 66, 68, 69, 70, 71, 72, 4, 8, 9, 10, 12, 13, 14, 36, 41, 46, 50, 51, 55, 57, 59, 60, 62, 63, 65, 66, 67, 68, 70, 71, 72, 5, 8, 10, 11, 12, 13, 14, 37, 40, 48, 49, 51, 53, 54, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 5, 8, 10, 11, 12, 13, 14, 38, 42, 47, 50, 53, 55, 58, 60, 61, 63, 64, 66, 67, 68, 69, 70, 71, 72, 9, 12, 14, 39, 54, 56, 59, 65, 66, 68, 70, 71, 72, 10, 12, 13, 14, 40, 54, 57, 59, 60, 65, 66, 67, 68, 70, 71, 72, 10, 12, 13, 14, 41, 55, 57, 60, 66, 67, 68, 70, 71, 72, 11, 13, 14, 42, 55, 58, 61, 67, 69, 71, 72, 43, 59, 60, 65, 66, 67, 68, 70, 71, 72, 44, 61, 67, 69, 71, 72, 45, 56, 59, 62, 65, 66, 68, 70, 71, 72, 46, 57, 59, 63, 65, 66, 67, 68, 70, 71, 72, 47, 58, 60, 63, 66, 67, 68, 70, 71, 72, 48, 57, 60, 61, 64, 66, 67, 68, 69, 70, 71, 72, 0, 1, 2, 3, 6, 8, 9, 10, 12, 13, 14, 49, 54, 62, 64, 65, 66, 68, 69, 70, 71, 72, 0, 1, 2, 3, 7, 8, 11, 12, 14, 50, 55, 63, 67, 68, 70, 71, 72, 51, 59, 60, 62, 63, 65, 66, 67, 68, 70, 71, 72, 52, 59, 64, 65, 66, 68, 69, 70, 71, 72, 53, 60, 61, 63, 64, 66, 67, 68, 69, 70, 71, 72, 3, 12, 14, 54, 65, 66, 70, 71, 72, 3, 13, 14, 55, 67, 71, 72, 56, 65, 70, 72, 57, 66, 67, 70, 71, 72, 58, 67, 71, 72, 59, 65, 66, 68, 70, 71, 72, 60, 66, 67, 68, 70, 71, 72, 61, 67, 69, 71, 72, 9, 12, 14, 62, 65, 68, 70, 71, 72, 11, 13, 14, 63, 67, 68, 70, 71, 72, 10, 12, 13, 14, 64, 66, 68, 69, 70, 71, 72, 65, 70, 72, 66, 70, 71, 72, 67, 71, 72, 3, 12, 13, 14, 68, 70, 71, 72, 3, 13, 14, 69, 71, 72, 14, 70, 72, 14, 71, 72, 72};
static const int WNUM[948] = {
    24, -24, 24, -24, -12, 12, -12, 12, 24, -24, -24, -24, 48, 24, -24, 24, -24, -24, -24, 24, 48, -24, 24, -36, 36, 12, -12, -24, 24, 12, -12, 8, -12, 12, 4, -12, 12, -4, 12, -12, -12, 24, 12, -12, -12, 12, 12, -24, 12, 24, -24, -24, 24, 12, -24, 12, -12, 12, -12, -12, 12, -12, 4, 24, -24, -24, -24, -24, -24, -24, 24, 24, 24, 48, 24, 24, 24, 72, 48, 48, -24, -24, -24, -144, -72, -24, -48, -48, -72, 72, 96, 48, 120, 24, -72, -72, 24, 24, -24, -24, -24, -24, -24, -24, 24, 24, 24, 24, 48, 24, 24, 72, 24, 72, -24, -24, -24, -72, -120, -48, -24, -96, -48, 24, 72, 120, 120, 24, -48, -96, 24, 12, -12, -24, -12, -24, 24, 12, 48, 12, 24, 12, 48, -12, -24, -24, -48, -48, -12, -24, -48, 12, 48, 48, 48, 24, -24, -48, 12, -12, 12, 12, 12, -24, -12, 12, 12, -24, -12, -12, -24, 12, 24, 24, 12, 24, 24, 12, 48, -12, -12, -12, -24, -72, -48, -24, -36, 36, 48, 36, 60, -36, -36, 12, -24, 24, 24, 24, -48, -24, 24, 24, -24, -24, -24, -24, -24, -24, 48, 72, 24, 24, 24, 48, 24, 24, 48, 24, -48, -24, -48, -144, -72, -24, -24, -24, -72, 72, 120, 48, 96, 24, -72, -72, 24, -12, 12, 12, 12, -12, -24, 12, 12, -24, -12, -12, -24, 24, 24, 12, 12, 24, 24, 12, 24, 24, -12, -12, -24, -12, -24, -72, -24, -12, -24, -24, 12, 48, 60, 48, 12, -24, -48, 12, -12, 12, 12, 12, -12, -24, 12, 12, -24, -12, -12, -24, 12, 48, 12, 24, 24, 12, 48, -24, -12, -24, -48, -72, -36, -24, 24, 96, 36, 24, -12, -60, 12, -24, 44, -44, -24, 24, 48, -48, -24, 24, 4, -12, -12, 12, 24, 12, 12, -12, -4, -4, -12, -24, -24, 12, 24, 12, 12, -12, -12, 4, -6, 11, -11, -6, 18, -18, 6, 1, -6, 12, 3, -4, -4, -12, 12, 3, -6, 1, 24, -24, -24, -24, -24, -24, 24, 24, 24, 96, 24, 24, 24, -72, -72, -24, -72, 72, 48, -24, 12, -24, -24, -12, 12, 24, 24, 24, 12, 24, -12, -48, -24, -24, -12, 24, 36, -12, 24, -24, -24, -24, -24, -24, 24, 24, 24, 72, 48, 24, 24, -48, -120, -48, -24, 24, 96, -24, 12, -12, -24, -24, 12, 48, 24, 24, 12, -36, -24, -12, -48, 36, 24, -12, 12, -12, -24, -24, 12, 24, 48, 12, 24, -12, -24, -36, -48, 24, 36, -12, 24, -24, -24, -24, -24, -24, 24, 72, 48, 24, 24, 48, -24, -72, -48, -72, -24, 48, 72, -24, 12, -12, -24, -24, 12, 24, 48, 12, 24, -24, -48, -24, -24, 12, 48, -12, -12, 24, -12, 12, -24, -24, -12, 24, 12, 12, 48, 24, -36, -48, -12, -24, 36, 24, -12, -12, 12, 12, -12, 12, -24, -24, -12, 12, 12, 24, 24, 48, -12, -48, -36, -24, 24, 36, -12, -6, 12, -6, 6, -24, -6, 12, 12, 36, -48, -12, 30, -6, 12, -24, -12, -24, 24, 24, 12, 12, 24, 24, -12, -24, -24, -48, -12, 24, 36, -12, -12, 12, 12, 12, -24, -12, 12, 12, -12, -12, -12, -24, 12, 12, 48, 12, 36, -36, -36, -36, -12, 36, 24, -12, -12, 12, 12, 12, -24, -12, 12, 12, -12, -12, -12, -24, 12, 12, 24, 24, 12, 36, -12, -24, -36, -48, 24, 36, -12, -24, 24, 24, 24, -24, -48, 24, 24, -24, -24, -24, -24, -24, 24, 24, 24, 48, 24, 24, 24, 48, -24, -72, -48, -72, -24, 48, 72, -24, -12, 12, 12, 12, -12, -24, 12, 12, -12, -12, -12, -24, 12, 12, 24, 24, 36, 12, -12, -60, -36, -12, 12, 48, -12, -12, 24, -12, 12, -12, -12, -24, 36, 24, 12, -36, -12, 12, -24, 24, 24, -24, 24, -24, -24, -24, -24, 24, 72, 24, 24, -48, -48, 24, -12, 12, 12, -12, 12, -12, -12, -24, 24, 36, 12, -12, -36, 12, -12, 24, -12, 12, -12, -12, -24, 60, 12, -48, 12, 12, -24, -24, 12, 24, 12, 24, -24, -24, 12, 3, -12, 12, 6, -12, 3, 12, -12, -24, -12, 36, 12, 24, -36, -12, 12, 12, -12, -24, -12, 12, 24, 12, 24, -24, -24, 12, 12, -12, -24, -12, 12, 36, 24, -12, -36, 12, 24, -24, -24, -24, -24, 48, 48, 24, 24, -24, -72, 24, -12, 18, 12, -30, -6, -18, 6, 12, 6, 12, -12, 6, -6, -6, -12, 6, 12, 12, 6, -12, -12, 6, -8, 8, 12, -20, -4, -12, 12, 12, -8, 4, -4, -12, 12, 12, -4, -12, 4, 24, -24, -24, -24, -24, 24, 24, 24, 72, -48, -48, 24, 12, -24, -24, 12, 24, 24, 12, -24, -24, 12, 24, -24, -24, -24, -24, 24, 48, 48, 24, -24, -72, 24, 12, -18, 18, 6, -6, -12, 12, 6, -6, 8, -12, 12, 4, -12, 12, -4, 4, -12, 12, -4, 12, -24, -12, 12, 24, -12, 4, -12, 12, -4, 24, -24, -24, -24, 48, 24, -24, 24, -24, -24, -24, 24, 48, -24, 12, -24, -12, 36, -12, -6, 12, -6, 6, -6, -12, 12, 6, -6, -12, 24, -12, 12, -12, -24, 12, 24, -12, -12, 12, 12, -12, 12, -12, -12, -12, 12, 24, -12, 6, -12, 6, 12, -12, -12, 12, 12, -24, 12, 12, -12, -12, 24, 12, -12, -12, 12, 3, -6, 6, 3, -6, 3, -4, 4, -4, -6, 6, -6, 1};
/* ---- end inlined perf/wtable.inc ---- */

static std::vector<int64> CV[NCORE];   /* one vector per core evaluation */
static std::vector<int64> WV[4];       /* the weight vectors: 1, d, A.d, d^2 */

static std::vector<int> tearr;         /* per CSR slot: |N(v) & N(adj[j])| */
static std::vector<int64> tdeg;        /* sum of degrees over that common set */
static std::vector<int64> a3e;         /* walks of length three across the edge */
static std::vector<int64> qe;          /* sum_{x in common set} cod(v, x) */
static std::vector<int64> k4e;         /* 4-cliques containing the edge */
static std::vector<int64> tri;         /* triangles at each node */
/* For each directed CSR slot j = (v -> u), the common neighbours of v and u,
   i.e. the third vertex of every triangle on that edge.  Built once and used
   wherever a loop would otherwise scan a whole adjacency list to rediscover
   them.  Total size is 6 * (number of triangles). */
static std::vector<int64> trioff;
static std::vector<int> trilist;
/* trislot[c] is the CSR slot of the edge (u, trilist[c]) for the entry c of the
   list belonging to edge (v,u) - i.e. the far edge of that triangle.  Recorded
   while the lists are built, where the slot is already in hand. */
static std::vector<int> trislot;   /* CSR slots fit in int32 for any graph this program can hold */
static bool have_trislot;
static bool have_trilists;
static std::vector<int64> k4cnt, k4dsum, k5cnt;

struct Scratch {
    std::vector<int> stamp, cb, pos, mk2;
    std::vector<int> touched, list2, cnt2, slot;
    std::vector<u64> lmask;
    void init(int N) {
        stamp.assign(N, -1); cb.assign(N, 0); pos.assign(N, -1); mk2.assign(N, -1);
        cnt2.assign(N, 0); slot.assign(N, -1);
    }
};

static inline int64 pc_and(const u64 *a, const u64 *b, size_t w) {
    int64 s = 0;
    for (size_t i = 0; i < w; i++) s += __builtin_popcountll(a[i] & b[i]);
    return s;
}
static inline int64 pc_and3(const u64 *a, const u64 *b, const u64 *c, size_t w) {
    int64 s = 0;
    for (size_t i = 0; i < w; i++) s += __builtin_popcountll(a[i] & b[i] & c[i]);
    return s;
}

static void par_for(int lo, int hi, int nth, const std::function<void(int, int, int)> &fn) {
    if (nth <= 1 || hi - lo < 512) { fn(0, lo, hi); return; }
    std::atomic<int> cursor(lo);
    const int CH = 32;
    std::vector<std::thread> th;
    for (int t = 0; t < nth; t++)
        th.emplace_back([&, t]() {
            for (;;) {
                int a = cursor.fetch_add(CH);
                if (a >= hi) break;
                fn(t, a, std::min(hi, a + CH));
            }
        });
    for (auto &x : th) x.join();
}

static inline int64 triple_common(int a, int b, int c) {
    if (use_bits)
        return pc_and3(&bits[(size_t)a * W], &bits[(size_t)b * W], &bits[(size_t)c * W], W);
    int s = a;
    if (deg[b] < deg[s]) s = b;
    if (deg[c] < deg[s]) s = c;
    int x = (s == a) ? b : a, y = (s == c) ? b : c;
    if (s == b) { x = a; y = c; }
    int64 cnt = 0;
    for (int64 j = off[s]; j < off[s + 1]; j++) {
        int v = adj[j];
        if (linked(v, x) && linked(v, y)) cnt++;
    }
    return cnt;
}

static int core_id(int k, int mask, int root, int w0, int w1 = 0, int w2 = 0,
                   int w3 = 0, int w4 = 0) {
    for (int i = 0; i < NCORE; i++) {
        const CoreSpec &c = CORES[i];
        if (c.k == k && c.mask == mask && c.root == root && c.w[0] == w0 &&
            c.w[1] == w1 && c.w[2] == w2 && c.w[3] == w3 && c.w[4] == w4) return i;
    }
    fprintf(stderr, "missing core %d/%d/%d\n", k, mask, root);
    exit(1);
}

/* ----------------------------------- pass 1: per-edge counts, and the lists

   Counting each edge's common neighbours and listing them are the same scan,
   but the lists cannot be placed until every count is known.  Rather than scan
   twice, the scan appends what it finds to a per-worker buffer and the buffer
   is copied into place once the offsets exist.  Workers take contiguous ranges
   of nodes - contiguous is what makes the copy one memcpy - chosen so that
   sum_{u in N(v)} d_u, which is what the scan costs, is about equal across
   them.                                                                      */
static void pass_edges(int nth, std::vector<Scratch> &SC) {
    const size_t E = adj.size();
    tearr.assign(E, 0);
    tdeg.assign(E, 0);
    tri.assign(n, 0);

    /* Work is still handed out by the atomic cursor, because a static split
       balances worse (see c18).  Each worker appends what it finds to its own
       buffer and records, per chunk it takes, where that chunk's entries begin;
       the copy then walks those records, each of which is contiguous in slot
       order because a chunk is a contiguous range of nodes. */
    std::vector<std::vector<std::pair<int, int>>> buf(nth);
    std::vector<std::vector<std::array<int64, 3>>> runs(nth);   /* slot, start, count */
    {
        std::atomic<int> cursor(0);
        const int CH = 32;
        auto body = [&](int t) {
            Scratch &s = SC[t];
            auto &b = buf[t];
            for (;;) {
                int lo = cursor.fetch_add(CH);
                if (lo >= n) break;
                const int hi = std::min(n, lo + CH);
                const int64 start = (int64)b.size();
                for (int v = lo; v < hi; v++) {
                    for (int64 j = off[v]; j < off[v + 1]; j++) s.stamp[adj[j]] = v;
                    int64 trisum = 0;
                    for (int64 j = off[v]; j < off[v + 1]; j++) {
                        int u = adj[j];
                        int c = 0; int64 ds = 0;
                        for (int64 k = off[u]; k < off[u + 1]; k++) {
                            int x = adj[k];
                            if (s.stamp[x] == v) { c++; ds += deg[x]; b.emplace_back(x, (int)k); }
                        }
                        tearr[j] = c; tdeg[j] = ds; trisum += c;
                    }
                    tri[v] = trisum / 2;
                }
                runs[t].push_back({off[lo], start, (int64)b.size() - start});
            }
        };
        if (nth <= 1) body(0);
        else {
            std::vector<std::thread> th;
            for (int t = 0; t < nth; t++) th.emplace_back(body, t);
            for (auto &x : th) x.join();
        }
    }

    trioff.assign(E + 1, 0);
    int64 total = 0;
    for (size_t j = 0; j < E; j++) { trioff[j] = total; total += tearr[j]; }
    trioff[E] = total;
    have_trilists = total <= (int64)200 * 1000 * 1000;
    if (!have_trilists) { trioff.clear(); trioff.shrink_to_fit(); return; }
    trilist.assign((size_t)total, 0);
    /* trislot pays for itself only where the lists are short: it replaces a
       d-length scan that finds nothing.  Where the average list is long the
       scan was never the cost and the extra array is pure memory traffic - it
       measured 0.938x at eight threads on a small-world graph.  One triangle
       per edge on average is the crossover used here. */
    have_trislot = total < (int64)E;
    if (have_trislot) trislot.assign((size_t)total, 0);
    for (int t = 0; t < nth; t++) {
        for (const auto &r : runs[t]) {
            int64 w = trioff[r[0]];
            for (int64 i = 0; i < r[2]; i++) {
                const auto &e = buf[t][(size_t)(r[1] + i)];
                if (have_trislot) trislot[w] = e.second;
                trilist[w++] = e.first;
            }
        }
        buf[t].clear(); buf[t].shrink_to_fit();
        runs[t].clear(); runs[t].shrink_to_fit();
    }
}

/* -------------------------------------------- pass 2: q and 4-cliques/edge */
/* Per-edge quantities and the 4/5-clique census, in one traversal: both wanted
   the common neighbourhood of each triangle, and it is built once. */
static void pass_edge2(int nth, std::vector<Scratch> &SC) {
    qe.assign(adj.size(), 0);
    k4e.assign(adj.size(), 0);
    k4cnt.assign(n, 0); k4dsum.assign(n, 0); k5cnt.assign(n, 0);
    const int c511_3 = core_id(5, 511, 3, 0), c1023 = core_id(5, 1023, 0, 0);
    std::vector<std::vector<int64>> A(nth), B(nth), C4(nth), D(nth);
    for (int t = 0; t < nth; t++) {
        A[t].assign(n, 0); B[t].assign(n, 0); C4[t].assign(n, 0); D[t].assign(n, 0);
    }
    par_for(0, n, nth, [&](int t, int lo, int hi) {
        Scratch &s = SC[t];
        std::vector<int64> &K4 = A[t], &KD = B[t], &K5 = C4[t], &T3 = D[t];
        std::vector<int> S;
        for (int v = lo; v < hi; v++) {
            const int64 b0 = off[v], b1 = off[v + 1];
            for (int64 j = b0; j < b1; j++) {
                s.stamp[adj[j]] = v; s.cb[adj[j]] = tearr[j]; s.slot[adj[j]] = (int)j;
            }
            for (int64 j = b0; j < b1; j++) {
                int u = adj[j];
                int64 q = 0, kk = 0;
                if (have_trilists) {
                    /* C = N(v) & N(u) is already listed; the 4-cliques on the
                       edge are the edges inside C.  That count is a property of
                       the undirected edge, so it is computed for one direction
                       and stored for both. */
                    for (int64 c = trioff[j]; c < trioff[j + 1]; c++) s.mk2[trilist[c]] = v;
                    const bool first = u > v;
                    for (int64 c = trioff[j]; c < trioff[j + 1]; c++) {
                        int w = trilist[c];
                        q += s.cb[w];
                        if (!first) continue;
                        /* S is the common neighbourhood of the triangle (v,u,w):
                           the entries of (v,w)'s triangle list that are also in
                           C.  The 4-cliques on the edge (v,u) are its members
                           above w, summed over w; the clique pass wanted the
                           same set, so it is built once here. */
                        const int64 jw = s.slot[w];
                        if (w <= u) {          /* only the clique count is wanted */
                            for (int64 e2 = trioff[jw]; e2 < trioff[jw + 1]; e2++) {
                                int x = trilist[e2];
                                if (x > w && s.mk2[x] == v) kk++;
                            }
                            continue;
                        }
                        S.clear();
                        for (int64 e2 = trioff[jw]; e2 < trioff[jw + 1]; e2++) {
                            int x = trilist[e2];
                            if (s.mk2[x] == v) { S.push_back(x); if (x > w) kk++; }
                        }
                        if (S.empty()) continue;
                        const int64 sz = (int64)S.size();
                        for (int x : S) T3[x] += 6 * sz;
                        for (size_t a1 = 0; a1 < S.size(); a1++) {
                            int x = S[a1];
                            if (x <= w) continue;
                            int qd[4] = {v, u, w, x};
                            int64 ds = (int64)deg[v] + deg[u] + deg[w] + deg[x];
                            for (int i = 0; i < 4; i++) { K4[qd[i]]++; KD[qd[i]] += ds - deg[qd[i]]; }
                            for (size_t a2 = a1 + 1; a2 < S.size(); a2++) {
                                int y = S[a2];
                                if (y <= x || !linked(y, x)) continue;
                                K5[v]++; K5[u]++; K5[w]++; K5[x]++; K5[y]++;
                            }
                        }
                    }
                    if (first) { k4e[j] = kk; k4e[rev[j]] = kk; }
                    for (int64 c = trioff[j]; c < trioff[j + 1]; c++) s.mk2[trilist[c]] = -1;
                    qe[j] = q;
                    continue;
                } else {
                    for (int64 k = off[u]; k < off[u + 1]; k++) {
                        int w = adj[k];
                        if (s.stamp[w] != v) continue;
                        q += s.cb[w];
                        for (int64 l = off[w]; l < off[w + 1]; l++) {
                            int x = adj[l];
                            if (x > w && s.stamp[x] == v && linked(x, u)) kk++;
                        }
                    }
                }
                qe[j] = q; k4e[j] = kk;
            }
            for (int64 j = b0; j < b1; j++) s.cb[adj[j]] = 0;
        }
    });
    CV[c511_3].assign(n, 0); CV[c1023].assign(n, 0);
    for (int t = 0; t < nth; t++)
        for (int v = 0; v < n; v++) {
            k4cnt[v] += A[t][v]; k4dsum[v] += B[t][v]; k5cnt[v] += C4[t][v];
            CV[c511_3][v] += D[t][v];
        }
    for (int v = 0; v < n; v++) CV[c1023][v] = 24 * k5cnt[v];
    {
        int a = core_id(4, 63, 0, 0), b = core_id(4, 63, 0, 1), c = core_id(4, 63, 1, 1);
        CV[a].assign(n, 0); CV[b].assign(n, 0); CV[c].assign(n, 0);
        for (int v = 0; v < n; v++) {
            CV[a][v] = 6 * k4cnt[v];
            CV[b][v] = (int64)deg[v] * 6 * k4cnt[v];
            CV[c][v] = 2 * k4dsum[v];
        }
    }
}

/* ------------------------------------------------------- pass 3: 4/5-cliques */
static void pass_cliques(int nth, std::vector<Scratch> &SC) {
    k4cnt.assign(n, 0); k4dsum.assign(n, 0); k5cnt.assign(n, 0);
    const int c511_3 = core_id(5, 511, 3, 0), c1023 = core_id(5, 1023, 0, 0);
    std::vector<std::vector<int64>> A(nth), B(nth), C(nth), D(nth);
    for (int t = 0; t < nth; t++) {
        A[t].assign(n, 0); B[t].assign(n, 0); C[t].assign(n, 0); D[t].assign(n, 0);
    }
    par_for(0, n, nth, [&](int t, int lo, int hi) {
        Scratch &s = SC[t];
        std::vector<int64> &K4 = A[t], &KD = B[t], &K5 = C[t], &T3 = D[t];
        std::vector<int> S;
        for (int v = lo; v < hi; v++) {
            for (int64 j = off[v]; j < off[v + 1]; j++) {
                s.stamp[adj[j]] = v; s.slot[adj[j]] = (int)j;
            }
            for (int64 j = off[v]; j < off[v + 1]; j++) {
                int u = adj[j];
                if (u <= v) continue;
                /* The triangles on (v,u) are its triangle list, so neither the
                   triangle search nor the search for a fourth vertex has to
                   walk a whole adjacency list. */
                if (have_trilists)
                    for (int64 c = trioff[j]; c < trioff[j + 1]; c++) s.mk2[trilist[c]] = v;
                const int64 cw0 = have_trilists ? trioff[j] : off[u];
                const int64 cw1 = have_trilists ? trioff[j + 1] : off[u + 1];
                for (int64 k = cw0; k < cw1; k++) {
                    int w = have_trilists ? trilist[k] : adj[k];
                    if (w <= u) continue;
                    if (!have_trilists && s.stamp[w] != v) continue;                    S.clear();
                    if (have_trilists) {
                        const int64 jw = s.slot[w];
                        for (int64 l = trioff[jw]; l < trioff[jw + 1]; l++) {
                            int x = trilist[l];
                            if (s.mk2[x] == v) S.push_back(x);
                        }
                    } else {
                        for (int64 l = off[w]; l < off[w + 1]; l++) {
                            int x = adj[l];
                            if (s.stamp[x] == v && linked(x, u)) S.push_back(x);
                        }
                    }
                    const int64 sz = (int64)S.size();
                    for (int x : S) T3[x] += 6 * sz;
                    for (size_t a1 = 0; a1 < S.size(); a1++) {
                        int x = S[a1];
                        if (x <= w) continue;
                        int q[4] = {v, u, w, x};
                        int64 ds = (int64)deg[v] + deg[u] + deg[w] + deg[x];
                        for (int i = 0; i < 4; i++) { K4[q[i]]++; KD[q[i]] += ds - deg[q[i]]; }
                        for (size_t a2 = a1 + 1; a2 < S.size(); a2++) {
                            int y = S[a2];
                            if (y <= x || !linked(y, x)) continue;
                            K5[v]++; K5[u]++; K5[w]++; K5[x]++; K5[y]++;
                        }
                    }
                }
                if (have_trilists)
                    for (int64 c = trioff[j]; c < trioff[j + 1]; c++) s.mk2[trilist[c]] = -1;
            }
        }
    });
    CV[c511_3].assign(n, 0); CV[c1023].assign(n, 0);
    for (int t = 0; t < nth; t++)
        for (int v = 0; v < n; v++) {
            k4cnt[v] += A[t][v]; k4dsum[v] += B[t][v]; k5cnt[v] += C[t][v];
            CV[c511_3][v] += D[t][v];
        }
    for (int v = 0; v < n; v++) CV[c1023][v] = 24 * k5cnt[v];
    /* the three weighted K4 cores */
    int a = core_id(4, 63, 0, 0), b = core_id(4, 63, 0, 1), c = core_id(4, 63, 1, 1);
    CV[a].assign(n, 0); CV[b].assign(n, 0); CV[c].assign(n, 0);
    for (int v = 0; v < n; v++) {
        CV[a][v] = 6 * k4cnt[v];
        CV[b][v] = (int64)deg[v] * 6 * k4cnt[v];
        CV[c][v] = 2 * k4dsum[v];
    }
}

/* ------------------------------------------- pass 4: everything local to a node

   For each v: the codegree ball, then the local adjacency masks over N(v),
   then a walk over the triangles at v.  Every core evaluation rooted at v is
   accumulated from those three structures.                                   */
static void pass_local(int nth, std::vector<Scratch> &SC,
                       std::vector<std::vector<int64>> &S126) {
    a3e.assign(adj.size(), 0);
    const int c220 = core_id(5, 220, 0, 0), c126_0 = core_id(5, 126, 0, 0),
              c221_2 = core_id(5, 221, 2, 0), c221_4 = core_id(5, 221, 4, 0),
              c127_2 = core_id(5, 127, 2, 0), c223_0 = core_id(5, 223, 0, 0),
              c223_1 = core_id(5, 223, 1, 0), c223_3 = core_id(5, 223, 3, 0),
              c254_0 = core_id(5, 254, 0, 0),
              c255_2 = core_id(5, 255, 2, 0), c255_4 = core_id(5, 255, 4, 0),
              c495_0 = core_id(5, 495, 0, 0),
              c511_0 = core_id(5, 511, 0, 0),
              d30a = core_id(4, 30, 0, 0), d30b = core_id(4, 30, 0, 1),
              d30c = core_id(4, 30, 1, 1), d30d = core_id(4, 30, 2, 1),
              d31a = core_id(4, 31, 0, 0), d31b = core_id(4, 31, 0, 0, 0, 1),
              d31c = core_id(4, 31, 0, 1), d31d = core_id(4, 31, 1, 1),
              d31e = core_id(4, 31, 2, 0), d31f = core_id(4, 31, 2, 0, 0, 1),
              d31g = core_id(4, 31, 2, 1), d31h = core_id(4, 31, 3, 0, 0, 1),
              t7a = core_id(3, 7, 0, 0), t7b = core_id(3, 7, 0, 1),
              t7c = core_id(3, 7, 0, 1, 1), t7d = core_id(3, 7, 0, 2),
              t7e = core_id(3, 7, 0, 3), t7f = core_id(3, 7, 1, 1),
              t7g = core_id(3, 7, 1, 2), t7h = core_id(3, 7, 1, 3),
              t7i = core_id(3, 7, 2, 1, 1);

    par_for(0, n, nth, [&](int t, int lo, int hi) {
        Scratch &s = SC[t];
        std::vector<int64> &sc126 = S126[t];
        for (int v = lo; v < hi; v++) {
            const int64 b0 = off[v], b1 = off[v + 1];
            const int dv = (int)(b1 - b0);
            if (dv == 0) continue;

            /* ---- codegree ball ---- */
            s.touched.clear();
            for (int64 j = b0; j < b1; j++) s.stamp[adj[j]] = v;
            for (int64 j = b0; j < b1; j++) {
                int u = adj[j];
                for (int64 k = off[u]; k < off[u + 1]; k++) {
                    int w = adj[k];
                    if (s.cb[w] == 0) s.touched.push_back(w);
                    s.cb[w]++;
                }
            }
            int64 p2 = 0, p3 = 0, c4_opp = 0, c5 = 0;
            /* The five-cycle term walks the adjacency list of every node in the
               distance-2 ball, which is half the run on a sparse graph.  The
               ball is discovered in the order the neighbourhoods were scanned,
               which is scattered; sorting it first makes both the CSR walk and
               the codegree gather run forward through memory. */
            for (size_t ti = 0; ti < s.touched.size(); ti++) {
                int w = s.touched[ti];
                int64 c = s.cb[w];
                p2 += c * c; p3 += c * c * c;
                c4_opp += (int64)deg[w] * c * c;
                if (ti + 1 < s.touched.size())
                    __builtin_prefetch(&adj[off[s.touched[ti + 1]]]);
#ifndef ORBIT_NO_C5
                int64 inner = 0;
                for (int64 k = off[w]; k < off[w + 1]; k++) inner += s.cb[adj[k]];
                c5 += c * inner;
#endif
            }
            CV[c220][v] = c5;
            CV[c126_0][v] = p3;
            CV[d30a][v] = p2;
            CV[d30b][v] = (int64)deg[v] * p2;
            CV[d30c][v] = c4_opp;

            /* ---- walks of length three across v's edges, and the sums over
                    them that need no triangle structure ---- */
            /* Three quantities are sums over the same (u in N(v), w in N(u))
               pairs once the ball is complete - the three-walk count on each of
               v's edges, the house rooted at its path middle, and the scatter
               for K(2,3) rooted on the degree-2 side.  One walk serves all
               three; they were three. */
            /* The degree-weighted codegree sum that the C4 core rooted at a
               side vertex wants is sum_w cbd[w] cb[w] with
               cbd[w] = sum_{u in N(v) & N(w)} d_u.  Reordering the sum makes it
               sum_{u in N(v)} d_u * (sum_{w in N(u)} cb[w]), and the inner sum
               is the three-walk count computed just below - so the whole cbd
               array, a second random-write stream through the ball build and a
               second array to clear, is unnecessary. */
            int64 h221_2 = 0, c4_side = 0;
            for (int64 j = b0; j < b1; j++) {
                int u = adj[j];
                int64 acc = 0, hh = 0, sq = 0;
                for (int64 k = off[u]; k < off[u + 1]; k++) {
                    const int64 c = s.cb[adj[k]];
                    acc += c;
                    hh += c * tearr[k];
                    sq += c * c;
                }
                a3e[j] = acc;
                h221_2 += hh;
                sc126[u] += sq;
                c4_side += (int64)deg[u] * acc;
            }
            CV[c221_2][v] = h221_2;
            CV[d30d][v] = c4_side;

            /* ---- the one triangle-walking sum that needs the ball: K(2,3)
                    with a rim edge, rooted on the degree-3 side.  Done here so
                    the second sweep does not have to rebuild the ball. ---- */
            {
                int64 g = 0;
                for (int64 j = b0; j < b1; j++) {
                    int u = adj[j];
                    for (int64 k = off[u]; k < off[u + 1]; k++) {
                        int w = adj[k];
                        if (s.stamp[w] != v) continue;
                        if (w < u) continue;   /* symmetric in (u,w); doubled */
                        if (have_trilists) {
                            for (int64 l = trioff[k]; l < trioff[k + 1]; l++)
                                g += 2 * s.cb[trilist[l]];
                        } else {
                            for (int64 l = off[u]; l < off[u + 1]; l++) {
                                int y = adj[l];
                                if (y != w && linked(y, w)) g += 2 * s.cb[y];
                            }
                        }
                    }
                }
                CV[c254_0][v] = g;
            }

            for (int w : s.touched) s.cb[w] = 0;
        }
    });

    /* A second sweep: the triangle work reads a3e for edges that belong to
       other nodes, so it cannot share a pass with the sweep that fills it. */
    par_for(0, n, nth, [&](int t, int lo, int hi) {
        Scratch &s = SC[t];
        std::vector<int> C;
        for (int v = lo; v < hi; v++) {
            const int64 b0 = off[v], b1 = off[v + 1];
            const int dv = (int)(b1 - b0);
            if (dv == 0) continue;
            /* Only cod(v, x) for x in N(v) is needed here, and that is the
               per-edge count already computed; the full distance-2 ball does
               not have to be rebuilt. */
            s.touched.clear();
            for (int64 j = b0; j < b1; j++) {
                s.stamp[adj[j]] = v;
                s.cb[adj[j]] = tearr[j];
                s.slot[adj[j]] = (int)j;
                s.touched.push_back(adj[j]);
            }

            /* ---- local adjacency over N(v): masks, or the triangle lists ----

               The masks answer |N(a) & N(b) & N(v)| in d/64 words but cost
               sum_{a in N(v)} d_a to build - and on a sparse graph they are
               almost entirely zero.  The same quantity is |trilist(a,b) & N(v)|,
               which costs t_ab, so when the graph is sparse enough the masks are
               not built at all.  Same gate as the wheel core below. */
            const size_t LW = (size_t)((dv + 63) / 64);
            int64 est_lists = 0;
            for (int64 j = b0; j < b1; j++) est_lists += (int64)tearr[j] * tearr[j];
            const int64 est_masks = (int64)dv * dv * (int64)LW;
            const bool use_lists = have_trilists && est_lists * 3 < est_masks;
            if (!use_lists) {
                if (s.lmask.size() < LW * (size_t)dv) s.lmask.assign(LW * (size_t)dv, 0);
                else std::fill(s.lmask.begin(), s.lmask.begin() + LW * (size_t)dv, 0ull);
                for (int i = 0; i < dv; i++) s.pos[adj[b0 + i]] = i;
                for (int i = 0; i < dv; i++) {
                    int a = adj[b0 + i];
                    u64 *row = &s.lmask[(size_t)i * LW];
                    for (int64 k = off[a]; k < off[a + 1]; k++)
                        if (s.stamp[adj[k]] == v) { int p = s.pos[adj[k]]; row[p >> 6] |= 1ull << (p & 63); }
                }
            }
            /* The wheel rooted at its hub is sum over all ordered pairs from
               N(v) of |N(a) & N(b) & N(v)|^2 - a C4-homomorphism count inside
               the local graph on N(v), whose adjacency lists are exactly the
               triangle lists of v's edges.  Two ways to evaluate it:

                 masks: every one of the d^2 pairs, d/64 words each;
                 lists: sum_{c in N(v)} t(v,c)^2, i.e. proportional to how many
                        triangles v is actually in.

               On a sparse graph the local graph is nearly empty and the mask
               form spends all its time confirming zeros; on a clustered one the
               lists are long and the mask form wins.  Both give the same
               number, so the cheaper estimate is taken per node. */
            int64 w495 = 0;
            if (use_lists) {
                for (int64 j = b0; j < b1; j++) {
                    s.list2.clear();
                    for (int64 c = trioff[j]; c < trioff[j + 1]; c++) {
                        int cc = trilist[c];
                        const int64 sc = s.slot[cc];
                        for (int64 e2 = trioff[sc]; e2 < trioff[sc + 1]; e2++) {
                            int b = trilist[e2];
                            if (s.cnt2[b] == 0) s.list2.push_back(b);
                            s.cnt2[b]++;
                        }
                    }
                    for (int b : s.list2) { int64 c2 = s.cnt2[b]; w495 += c2 * c2; s.cnt2[b] = 0; }
                }
            } else {
                for (int i = 0; i < dv; i++) {
                    const u64 *ri = &s.lmask[(size_t)i * LW];
                    for (int j = 0; j < dv; j++) {
                        int64 x = pc_and(ri, &s.lmask[(size_t)j * LW], LW);
                        w495 += x * x;
                    }
                }
            }
            CV[c495_0][v] = w495;

            /* ---- triangles at v ---- */
            int64 T0 = 0, T1 = 0, T2 = 0, T3 = 0, T4 = 0;
            int64 g221_4 = 0, g127_2 = 0, g223_0 = 0, g223_1 = 0, g223_3 = 0,
                  g255_4 = 0, g255_2 = 0, g511_0 = 0, dia_c = 0, dia_d = 0, dia_x = 0;
            for (int64 j = b0; j < b1; j++) {
                int u = adj[j];
                const u64 *ru = use_lists ? nullptr : &s.lmask[(size_t)s.pos[u] * LW];
                int64 inner_cod = 0;
                C.clear();
                for (int64 k = off[u]; k < off[u + 1]; k++) {
                    int w = adj[k];
                    if (s.stamp[w] != v) continue;
                    C.push_back(w);
                    const int64 cuw = tearr[k];
                    T0 += 1;
                    T1 += deg[u];
                    T2 += WV[2][u];
                    T3 += (int64)deg[u] * deg[u];
                    T4 += (int64)deg[u] * deg[w];
                    g221_4 += a3e[k];
                    g127_2 += cuw * cuw;
                    g223_0 += (int64)s.cb[u] * s.cb[w];
                    g223_3 += qe[k];
                    g255_4 += 2 * k4e[k];
                    inner_cod += cuw;
                    dia_c += cuw;
                    dia_d += tdeg[k];
                    dia_x += (int64)deg[u] * cuw;
                    /* The local codegree and the far edge's list are the same
                       for (u,w) and (w,u), and both quantities below are
                       symmetric in them, so one ordering is taken and doubled. */
                    if (w > u) {
                        int64 tvv;
                        if (use_lists) {
                            tvv = 0;
                            for (int64 l = trioff[k]; l < trioff[k + 1]; l++)
                                if (s.stamp[trilist[l]] == v) tvv++;
                        } else {
                            tvv = pc_and(ru, &s.lmask[(size_t)s.pos[w] * LW], LW);
                        }
                        g255_2 += 2 * tvv * cuw;
                        g511_0 += 2 * tvv * tvv;
                    }
                }
                g223_1 += (int64)s.cb[u] * inner_cod;

            }
            CV[t7a][v] = T0;
            CV[t7b][v] = (int64)deg[v] * T0;
            CV[t7c][v] = (int64)deg[v] * T1;
            CV[t7d][v] = WV[2][v] * T0;
            CV[t7e][v] = (int64)deg[v] * deg[v] * T0;
            CV[t7f][v] = T1;
            CV[t7g][v] = T2;
            CV[t7h][v] = T3;
            CV[t7i][v] = T4;
            CV[c221_4][v] = g221_4;
            CV[c127_2][v] = g127_2;
            CV[c223_0][v] = g223_0;
            CV[c223_1][v] = g223_1;
            CV[c223_3][v] = g223_3;
            CV[c255_4][v] = g255_4;
            CV[c255_2][v] = g255_2;
            CV[c511_0][v] = g511_0;
            CV[d31e][v] = dia_c;
            CV[d31f][v] = (int64)deg[v] * dia_c;
            CV[d31g][v] = dia_x;
            CV[d31h][v] = dia_d;

            /* ---- diamond cores rooted on the degree-3 side ---- */
            int64 e2 = 0, ed = 0, edeg = 0;
            for (int64 j = b0; j < b1; j++) {
                int64 c = tearr[j];
                e2 += c * c; ed += c * tdeg[j]; edeg += (int64)deg[adj[j]] * c * c;
            }
            CV[d31a][v] = e2;
            CV[d31b][v] = ed;
            CV[d31c][v] = (int64)deg[v] * e2;
            CV[d31d][v] = edeg;

            for (int w : s.touched) s.cb[w] = 0;
            if (!use_lists) for (int i = 0; i < dv; i++) s.pos[adj[b0 + i]] = -1;
        }
    });
}

/* --------- pass 5: the K(2,3)-plus-rim-edge core rooted at the spare vertex

   E(x0, x1) counts ordered adjacent pairs inside N(x0) & N(x1).  Building it
   for one x0 at a time and scattering to x0's neighbours keeps the whole thing
   to sum_e t_e^2 plus sum_v d_v^2 work and needs no table of pairs.          */
static void pass_e254(int nth, std::vector<Scratch> &SC,
                      std::vector<std::vector<int64>> &S254) {
    par_for(0, n, nth, [&](int t, int lo, int hi) {
        Scratch &s = SC[t];
        std::vector<int64> &acc = S254[t];
        std::vector<int> touched, list2, cnt2, slot;
        for (int x0 = lo; x0 < hi; x0++) {
            const int64 b0 = off[x0], b1 = off[x0 + 1];
            if (b1 == b0) continue;
            for (int64 j = b0; j < b1; j++) s.stamp[adj[j]] = x0;
            touched.clear();
            for (int64 j = b0; j < b1; j++) {
                int y = adj[j];
                /* the ordered pairs (y,z) inside N(x0) with y~z are exactly the
                   triangles on the edge (x0,y), and the far edge's slot is
                   recorded alongside, so neither has to be searched for */
                const bool fast = have_trilists && have_trislot;
                const int64 z0 = fast ? trioff[j] : off[y];
                const int64 z1 = fast ? trioff[j + 1] : off[y + 1];
                for (int64 kk = z0; kk < z1; kk++) {
                    const int z = fast ? trilist[kk] : adj[kk];
                    const int64 k = fast ? (int64)trislot[kk] : kk;
                    if (!fast && s.stamp[z] != x0) continue;
                    /* (y,z) and (z,y) have the same common neighbourhood, so
                       one order is walked and counted twice */
                    if (z < y) continue;
                    /* the vertices adjacent to both y and z are exactly the
                       triangle list of the edge (y,z), whose slot is k */
                    if (have_trilists) {
                        for (int64 l = trioff[k]; l < trioff[k + 1]; l++) {
                            int x1 = trilist[l];
                            if (s.cb[x1] == 0) touched.push_back(x1);
                            s.cb[x1] += 2;
                        }
                    } else {
                        for (int64 l = off[y]; l < off[y + 1]; l++) {
                            int x1 = adj[l];
                            if (x1 == z || !linked(x1, z)) continue;
                            if (s.cb[x1] == 0) touched.push_back(x1);
                            s.cb[x1] += 2;
                        }
                    }
                }
            }
            /* If x0 is in no triangle the accumulated set is empty and every
               one of these sums is zero - but they still cost sum_{v in N(x0)}
               d_v to compute.  On a sparse graph that is most of the nodes. */
            if (!touched.empty()) {
                for (int64 j = b0; j < b1; j++) {
                    int v = adj[j];
                    int64 sum = 0;
                    for (int64 k = off[v]; k < off[v + 1]; k++) sum += s.cb[adj[k]];
                    acc[v] += sum;
                }
                for (int x : touched) s.cb[x] = 0;
            }
        }
    });
}


/* ------------------ pass 6: the two cores that are sums over pairs from C

   For an edge (v,u) let C = N(v) & N(u).  Both the wheel rooted at a rim vertex
   and the K(2,3)-plus-rim-edge core rooted inside the extra edge are sums over
   pairs drawn from C, and both follow from counting how many members of C each
   vertex sees:  sum over pairs from C  =  sum_y |N(y) & C|^2.

   That count is a property of the *undirected* edge, so it is built once here
   and used for both endpoints, halving the sum_e t_e d work the per-direction
   form was doing.                                                            */
static void pass_pairs(int nth, std::vector<Scratch> &SC,
                       std::vector<std::vector<int64>> &A2, std::vector<std::vector<int64>> &A5) {
    par_for(0, n, nth, [&](int t, int lo, int hi) {
        Scratch &s = SC[t];
        std::vector<int64> &acc254 = A2[t], &acc495 = A5[t];
        for (int v = lo; v < hi; v++) {
            for (int64 j = off[v]; j < off[v + 1]; j++) {
                int u = adj[j];
                if (u <= v) continue;
                s.list2.clear();
                if (have_trilists) {
                    for (int64 c = trioff[j]; c < trioff[j + 1]; c++) {
                        int x0 = trilist[c];
                        for (int64 k = off[x0]; k < off[x0 + 1]; k++) {
                            int y = adj[k];
                            if (s.cnt2[y] == 0) s.list2.push_back(y);
                            s.cnt2[y]++;
                        }
                    }
                } else {
                    for (int64 kk = off[u]; kk < off[u + 1]; kk++) s.stamp[adj[kk]] = n + v;
                    for (int64 kk = off[v]; kk < off[v + 1]; kk++) {
                        int x0 = adj[kk];
                        if (s.stamp[x0] != n + v) continue;
                        for (int64 k = off[x0]; k < off[x0 + 1]; k++) {
                            int y = adj[k];
                            if (s.cnt2[y] == 0) s.list2.push_back(y);
                            s.cnt2[y]++;
                        }
                    }
                }
                if (s.list2.empty()) continue;   /* edge is in no triangle */
                int64 F = 0;
                for (int y : s.list2) { int64 c = s.cnt2[y]; F += c * c; }
                acc254[v] += F; acc254[u] += F;
                int64 gv = 0, gu = 0;
                for (int64 k = off[u]; k < off[u + 1]; k++) { int64 c = s.cnt2[adj[k]]; gv += c * c; }
                for (int64 k = off[v]; k < off[v + 1]; k++) { int64 c = s.cnt2[adj[k]]; gu += c * c; }
                acc495[v] += gv; acc495[u] += gu;
                for (int y : s.list2) s.cnt2[y] = 0;
            }
        }
    });
}

/* ------------------------------------------------------------- the driver */
/* The 73 programs share subtrees - every one that reaches a core through a
   pendant, for instance, applies the same adjacency multiplications - and the
   table is hash-consed so an identical subtree is one node.  Evaluating each
   node once turns roughly 300 sparse matrix-vector products into 74 node
   evaluations, of which only the A-nodes cost O(m). */
static std::vector<std::vector<int64>> NODEVAL;
static std::vector<char> NODEDONE;

static const std::vector<int64> &eval_node(int id) {
    if (NODEDONE[id]) return NODEVAL[id];
    const int *nd = NODES[id];
    if (nd[0] == 0) NODEVAL[id] = WV[0];
    else if (nd[0] == 3) NODEVAL[id] = CV[nd[1]];
    else if (nd[0] == 1) {
        const std::vector<int64> &x = eval_node(nd[1]);
        std::vector<int64> out(n, 0);
        for (int v = 0; v < n; v++) {
            int64 t = 0;
            for (int64 j = off[v]; j < off[v + 1]; j++) t += x[adj[j]];
            out[v] = t;
        }
        NODEVAL[id] = std::move(out);
    } else {
        const std::vector<int64> &a = eval_node(nd[1]);
        const std::vector<int64> &b = eval_node(nd[2]);
        std::vector<int64> out(n, 0);
        for (int v = 0; v < n; v++) out[v] = a[v] * b[v];
        NODEVAL[id] = std::move(out);
    }
    NODEDONE[id] = 1;
    return NODEVAL[id];
}

/* Per-pass wall times, for directing the investigation.  ORBIT_TIMING=1 prints
   them to stderr; nothing else in the program depends on this. */
static bool timing_on;
static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double t_mark;
static void tick(const char *what) {
    if (!timing_on) return;
    double t = now_s();
    fprintf(stderr, "  %-14s %7.3f s\n", what, t - t_mark);
    t_mark = t;
}

static void algebraic_all(int nth) {
    timing_on = getenv("ORBIT_TIMING") != 0;
    t_mark = now_s();
    if (nth < 1) nth = 1;
    std::vector<Scratch> SC(nth);
    for (auto &s : SC) s.init(n);

    WV[0].assign(n, 1);
    WV[1].assign(n, 0); WV[2].assign(n, 0); WV[3].assign(n, 0);
    for (int v = 0; v < n; v++) WV[1][v] = deg[v];
    for (int v = 0; v < n; v++) {
        int64 s = 0;
        for (int64 j = off[v]; j < off[v + 1]; j++) s += deg[adj[j]];
        WV[2][v] = s;
    }
    for (int v = 0; v < n; v++) WV[3][v] = (int64)deg[v] * deg[v];

    for (int i = 0; i < NCORE; i++) CV[i].assign(n, 0);
    std::vector<std::vector<int64>> S126(nth), S254(nth);
    for (int t = 0; t < nth; t++) { S126[t].assign(n, 0); S254[t].assign(n, 0); }

    tick("setup");
    pass_edges(nth, SC);
    tick("edges+lists");
    pass_edge2(nth, SC);
    tick("q+k4perEdge");
    pass_local(nth, SC, S126);
    tick("local");
    pass_e254(nth, SC, S254);
    tick("e254");
    std::vector<std::vector<int64>> P254(nth), P495(nth);
    for (int t = 0; t < nth; t++) { P254[t].assign(n, 0); P495[t].assign(n, 0); }
    pass_pairs(nth, SC, P254, P495);
    tick("pairs");
    {
        int a = core_id(5, 254, 2, 0), b = core_id(5, 495, 1, 0);
        for (int t = 0; t < nth; t++)
            for (int v = 0; v < n; v++) { CV[a][v] += P254[t][v]; CV[b][v] += P495[t][v]; }
    }

    {
        int c126_2 = core_id(5, 126, 2, 0), c254_4 = core_id(5, 254, 4, 0);
        for (int t = 0; t < nth; t++)
            for (int v = 0; v < n; v++) {
                CV[c126_2][v] += S126[t][v];
                CV[c254_4][v] += S254[t][v];
            }
    }
    {   /* cores that need nothing but the per-edge arrays */
        int c207_0 = core_id(5, 207, 0, 0), c207_1 = core_id(5, 207, 1, 0),
            c221_0 = core_id(5, 221, 0, 0), c127_0 = core_id(5, 127, 0, 0),
            c255_0 = core_id(5, 255, 0, 0);
        for (int v = 0; v < n; v++) {
            int64 a = 0, b = 0, c = 0, d = 0;
            for (int64 j = off[v]; j < off[v + 1]; j++) {
                int64 t = tearr[j];
                a += 2 * tri[adj[j]] * t;
                b += t * a3e[j];
                c += t * t * t;
                d += t * 2 * k4e[j];
            }
            CV[c207_0][v] = 4 * tri[v] * tri[v];
            CV[c207_1][v] = a;
            CV[c221_0][v] = b;
            CV[c127_0][v] = c;
            CV[c255_0][v] = d;
        }
    }

    /* homomorphism counts, then the induced columns */
    NODEVAL.assign(NNODE, {});
    NODEDONE.assign(NNODE, 0);
    std::vector<const std::vector<int64> *> Hv(73);
    for (int c = 0; c < 73; c++) Hv[c] = &eval_node(PROGROOT[c]);
    tick("hom-eval");
    if (getenv("ORBIT_DUMP_HOM")) {
        FILE *f = fopen(getenv("ORBIT_DUMP_HOM"), "w");
        for (int v = 0; v < n; v++) {
            for (int c = 0; c < 73; c++) fprintf(f, "%lld%c", (*Hv[c])[v], c == 72 ? '\n' : ' ');
        }
        fclose(f);
    }
    /* The transform is 948 multiply-adds per node.  128-bit accumulation is
       only needed when a homomorphism count is large enough that the widest
       row of the matrix could overflow 64 bits; that takes a degree in the
       thousands.  The bound is checked against the counts actually produced,
       so the narrow path is taken when it is provably safe and never
       otherwise. */
    int64 rowsum = 0;
    for (int c = 0; c < 73; c++) {
        int64 r = 0;
        for (int k = WOFF[c]; k < WOFF[c + 1]; k++) r += WNUM[k] < 0 ? -WNUM[k] : WNUM[k];
        if (r > rowsum) rowsum = r;
    }
    /* Every pattern here is connected on at most five vertices, so each of the
       four non-root images is placed along an edge from one already placed:
       no homomorphism count exceeds d_max^4.  That bound costs nothing to
       compute, where scanning the counts themselves costs a full pass over 73
       vectors - which measured 0.922x, more than the narrow path saves. */
    int dmax = 0;
    for (int v = 0; v < n; v++) if (deg[v] > dmax) dmax = deg[v];
    const __int128 maxh = (__int128)dmax * dmax * dmax * dmax;
    const bool narrow = maxh == 0 ||
        (__int128)rowsum * maxh <= (__int128)0x7fffffffffffffffLL;
    par_for(0, n, nth, [&](int, int lo, int hi) {
        if (narrow) {
            for (int v = lo; v < hi; v++)
                for (int c = 0; c < 73; c++) {
                    int64 acc = 0;
                    for (int k = WOFF[c]; k < WOFF[c + 1]; k++)
                        acc += (int64)WNUM[k] * (*Hv[WCOL[k]])[v];
                    orbit[(size_t)v * 73 + c] = acc / WDEN;
                }
        } else {
            for (int v = lo; v < hi; v++)
                for (int c = 0; c < 73; c++) {
                    __int128 acc = 0;
                    for (int k = WOFF[c]; k < WOFF[c + 1]; k++)
                        acc += (__int128)WNUM[k] * (*Hv[WCOL[k]])[v];
                    orbit[(size_t)v * 73 + c] = (int64)(acc / WDEN);
                }
        }
    });
    tick("transform");
}

static int thread_count() {
	return 1;  // force single-thread

    const char *e = getenv("ORBIT_THREADS");
    int t = e ? atoi(e) : 1;
    return t < 1 ? 1 : t;
}

static void count_all() {
    int nth = thread_count();
    if (nth > n) nth = n > 0 ? n : 1;
    if (nth == 1) {
        Ctx c;
        c.scratch.assign(6 * (size_t)n + 64, 0);
        c.acc = orbit.data();
        roots(c, 0, n);
        return;
    }
    const int CHUNK = 16;
    std::vector<std::vector<int64>> bufs(nth);
    std::atomic<int> cursor(0);
    std::vector<std::thread> th;
    for (int t = 0; t < nth; t++)
        th.emplace_back([&, t]() {
            Ctx c;
            c.scratch.assign(6 * (size_t)n + 64, 0);
            if (t == 0) c.acc = orbit.data();
            else { bufs[t].assign((size_t)n * 73, 0); c.acc = bufs[t].data(); }
            for (;;) {
                int lo = cursor.fetch_add(CHUNK);
                if (lo >= n) break;
                roots(c, lo, std::min(n, lo + CHUNK));
            }
        });
    for (auto &x : th) x.join();
    for (int t = 1; t < nth; t++) {
        const int64 *src = bufs[t].data();
        int64 *dst = orbit.data();
        for (size_t i = 0, e = (size_t)n * 73; i < e; i++) dst[i] += src[i];
        bufs[t].clear(); bufs[t].shrink_to_fit();
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input graph> <output name>\n", argv[0]);
        return 1;
    }
    read_graph(argv[1]);

    clock_t startTime, endTime;
	startTime = clock();

    build_tables();
    orbit.assign((size_t)n * 73, 0);

    algebraic_all(thread_count());

    if (getenv("ORBIT_ENUM")) {
        /* cross-check path: recompute the five-node columns by enumeration.
           Slow by design; it exists so the algebra can be diffed against a
           method that shares none of its machinery. */
        for (int v = 0; v < n; v++)
            for (int c = 15; c < 73; c++) orbit[(size_t)v * 73 + c] = 0;
        count_all();
    }

    /* graphlet totals: each pattern's count is the sum of any one of its orbit
       columns divided by that orbit's size */
    int64 total[29] = {0};
    {
        int done[29] = {0};
        for (int k = 3; k <= 5; k++)
            for (int c = 0; c < NCAN[k]; c++) {
                int col = CAN[k][c].tcol;
                int64 sum = 0;
                for (int v = 0; v < n; v++) sum += orbit[(size_t)v * 73 + col];
                total[CAN[k][c].g] = sum / CAN[k][c].tsize;
                done[CAN[k][c].g] = 1;
            }
        for (int i = 0; i < 29; i++) if (!done[i]) { fprintf(stderr, "graphlet %d unset\n", i); return 1; }
    }

	endTime = clock();
	printf("total: %.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);

    char path[4096];
    FILE *out = fopen(argv[2], "wb");
    if (!out) { perror(argv[2]); return 1; }
    snprintf(path, sizeof path, "%s.gr_freq", argv[2]);
    FILE *gr = fopen(path, "wb");
    snprintf(path, sizeof path, "%s.ndump2", argv[2]);
    FILE *nd = fopen(path, "wb");
    for (int i = 0; i < 29; i++) {
        fprintf(out, "%d\t%lld\n", i + 1, total[i]);
        if (gr) fprintf(gr, "%d\t%lld\n", i + 1, total[i]);
    }
    if (nd) {
        /* 73 numbers per node through fprintf is n*73 formatted calls, which is
           a visible share of the run now that the counting is fast.  Format
           into one buffer with a hand-rolled decimal conversion and hand it to
           the file in large blocks; the bytes produced are identical. */
        std::vector<char> buf;
        buf.reserve(1 << 20);
        char tmp[24];
        for (int v = 0; v < n; v++) {
            const int64 *row = &orbit[(size_t)v * 73];
            for (int i = 0; i < 73; i++) {
                if (i) buf.push_back(' ');
                int64 x = row[i];
                if (i == 0) x = deg[v];
                if (x == 0) { buf.push_back('0'); continue; }
                int k = 0;
                if (x < 0) { buf.push_back('-'); x = -x; }
                while (x) { tmp[k++] = (char)('0' + x % 10); x /= 10; }
                while (k) buf.push_back(tmp[--k]);
            }
            buf.push_back('\n');
            if (buf.size() >= (1u << 20)) {
                fwrite(buf.data(), 1, buf.size(), nd);
                buf.clear();
            }
        }
        if (!buf.empty()) fwrite(buf.data(), 1, buf.size(), nd);
    }
    fclose(out);
    if (gr) fclose(gr);
    if (nd) fclose(nd);
    return 0;
}
