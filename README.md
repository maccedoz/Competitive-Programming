# Técnicas de Programação
## 1. Lower/Upper Bound e Busca Binária
**O que faz:**
Busca binária (e funções `lower_bound`/`upper_bound`) localiza posições em um array ordenado
o O(log n).
**Quando usar (Tipo de problema):**
- Busca de valor específico: encontrar se um elemento existe.
- Busca de primeiro elemento ≥/≤ valor: determinar limites inferior e superior.
- Decisões monotônicas: otimização por teste de satisfabilidade em pesquisas paramétricas.
**Exemplo de código (C++):**
```cpp
#include <bits/stdc++.h>
using namespace std;
// Verifica se x existe em vetor ordenado a usando busca binária clássica
bool exists(const vector<int>& a, int x) {
  int l = 0, r = a.size() - 1; // limites inicial e final
  while (l <= r) { // enquanto houver espaço
    int m = l + (r - l) / 2; // ponto médio (evita overflow)
    if (a[m] == x) // elemento encontrado
      return true;
    if (a[m] < x) // buscar na metade direita
      l = m + 1;
    else // buscar na metade esquerda
      r = m - 1;
  }
  return false; // não encontrado
}

int main() {
vector<int> v = {1,3,5,7,9};
cout << (exists(v, 7) ? "Encontrado" : "Não encontrado");
}
```
---
## 2. Sliding Window
**O que faz:**
Mantém uma janela de limites móveis para processar subarrays/ substrings contíguas em O(n).
**Quando usar (Tipo de problema):**
- Subarrays contíguos com restrições cumulativas: soma, conta ou propriedades de caracteres.
- Máxima/mínima janela que satisfaz condição: soma ≤ K, número de elementos distintos, etc.
**Exemplo de código (C++):**
```cpp
// Retorna o tamanho máximo de subarray cuja soma é ≤ K
int maxSubarrayAtMostK(const vector<int>& a, int K) {
  int sum = 0;
  int l = 0;
  int best = 0;
  for (int r = 0; r < a.size(); ++r) {
    sum += a[r];
    while (sum > K) {
      sum -= a[l];
      l++;
    }
    best = max(best, r - l + 1);
  }
  return best;
}
```
---
## 3. Two Pointers
**O que faz:**
Usa dois índices para varrer um array (geralmente ordenado) em tempo linear.
**Quando usar (Tipo de problema):**
- Soma em dois elementos: encontrar pares com soma X.
- Remoção in-place: eliminar duplicados ou elementos que atendam condição.
- Intersecção/união de arrays ordenados.
**Exemplo de código (C++):**
```cpp
// Retorna todos pares (a[l], a[r]) cuja soma é X
vector<pair<int,int>> pairsWithSum(const vector<int>& a, int X) {
vector<pair<int,int>> res;
int l = 0, r = a.size() - 1;
while (l < r) {
int s = a[l] + a[r];
if (s == X) {
res.emplace_back(a[l], a[r]);
l++;
r--;
}
else if (s < X)
l++;
else
r--;
}
return res;
}
```
---
## 4. Prefix Sum
**O que faz:**
Pré-processa somas parciais para consultas de soma de intervalos em O(1).
**Quando usar (Tipo de problema):**
- Consultas repetidas de soma em subintervalos.
- Diferença de prefixos para obter soma i..j.
- Problemas de soma fixa em subarrays.
**Exemplo de código (C++):**
```cpp
vector<int> prefix(const vector<int>& a) {
vector<int> p(a.size() + 1, 0);
for (int i = 0; i < a.size(); ++i)
  p[i+1] = p[i] + a[i];
  return p;
}
// Soma de a[i..j]: p[j+1] - p[i]
```
---
## 5. Recursão
**O que faz:**
Divide problema em subproblemas auto-similares, chamando a si próprio.
**Quando usar (Tipo de problema):**
- Árvores, backtracking simples, cálculos matemáticos recursivos.
- Casos base claros e profundidade limitada.
**Exemplo de código (C++):**
```cpp
// Calcula o n-ésimo número de Fibonacci
int fib(int n) {
  if (n < 2) return n;
  return fib(n-1) + fib(n-2);
}
```
---
## 6. Backtracking
**O que faz:**
Explora espaço de estados gerando soluções parciais e retrocedendo (undo) escolhas.
**Quando usar (Tipo de problema):**
- Geração de permutações, combinações, quebra-cabeças (sudoku, n-rainhas).
- Busca em árvore de soluções com poda.
**Exemplo de código (C++):**
```cpp
void permute(vector<int>& a, int i, vector<vector<int>>& res) {
  if (i == a.size()) {
    res.push_back(a);
    return;
  }
  for (int j = i; j < a.size(); ++j) {
    swap(a[i], a[j]);
    permute(a, i + 1, res);
    swap(a[i], a[j]);
  }
}
```
---
## 7. Bitmask
**O que faz:**
Representa subconjuntos como bits de um inteiro, iterando em 0..2n−1.
**Quando usar (Tipo de problema):**
- Subconjuntos de elementos (n ≤ 20–25).
- Programação dinâmica sobre subconjuntos.
**Exemplo de código (C++):**
```cpp
int n = 4;
for (int mask = 0; mask < (1 << n); ++mask) {
  vector<int> subset;
  for (int i = 0; i < n; ++i) {
    if (mask & (1 << i))
    subset.push_back(i);
  }
}
```
---
## 8. Greedy (Guloso)
**O que faz:**
Faz escolha local ótima em cada passo, sem retornar.
**Quando usar (Tipo de problema):**
- Troco, interval scheduling, problemas com subestrutura ótima comprovada.
**Exemplo de código (C++):**
```cpp
vector<int> coins = {25, 10, 5, 1};
int change = 31;
vector<int> used;
for (int c : coins) {
  while (change >= c) {
    used.push_back(c);
    change -= c;
  }
}
```
---
## 9. Programação Dinâmica Clássica (DP)

**O que faz**  
Armazena resultados de subproblemas para evitar recomputação.

**Quando usar (Tipos de problema)**  
- Subproblemas sobrepostos  
- Subestrutura ótima (ex.: knapsack, LCS)

**Exemplo de código (C++)**  
```cpp
int knap(int W, const vector<int>& w, const vector<int>& v) {
    int n = w.size();
    // dp[i][j] = melhor valor usando itens [0..i-1] com capacidade j
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j <= W; ++j) {
            // opção 1: não pegar o item i-1
            int sem_item = dp[i - 1][j];

            // opção 2: pegar o item i-1 (se couber no peso j)
            int com_item = 0;
            if (j >= w[i - 1]) {
                com_item = dp[i - 1][j - w[i - 1]] + v[i - 1];
            }

            // escolhe a melhor das duas opções
            dp[i][j] = max(sem_item, com_item);
        }
    }

    return dp[n][W];
}
```
---

## Módulo Avançado B

### 10. Teoria dos Números

**O que faz**  
- Primalidade  
- Cálculo de GCD/LCM  
- Inverso modular  
- Resolução de equações diofantinas

**Quando usar (Tipos de problema)**  
- Criba de Eratóstenes para gerar primos  
- Simplificação de frações com GCD/LCM  
- Criptografia e congruências com inverso modular  
- Solução de equações diofantinas lineares

**Exemplo de código (C++)**  
```cpp
// GCD via algoritmo de Euclides
int gcd(int a, int b) {
    return b ? gcd(b, a % b) : a;
}

// Exponenciação rápida para modPow
int modPow(int base, int exp, int m) {
    long long result = 1, b = base;
    while (exp > 0) {
        if (exp & 1) result = (result * b) % m;
        b = (b * b) % m;
        exp >>= 1;
    }
    return result;
}

// Inverso modular usando Fermat (m primo)
int modinv(int a, int m) {
    return modPow(a, m - 2, m);
}

// Extenso algoritmo de Euclides para ax + by = gcd(a,b)
int extgcd(int a, int b, int &x, int &y) {
    if (b == 0) {
        x = 1; y = 0;
        return a;
    }
    int x1, y1;
    int g = extgcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return g;
}
11. LIS e DP Bitmask

O que faz

    LIS: encontra a maior subsequência estritamente crescente em O(n log n).

    DP Bitmask: resolve problemas sobre todos os subconjuntos (n ≤ 20–25) usando bits como estados.

Quando usar (Tipos de problema)

    LIS: padrões de crescimento em sequências (ex.: empilhamento de caixas).

    DP Bitmask: TSP, cobertura de conjuntos, atribuição de tarefas.

Exemplo de código (C++)

vector<int> lis(const vector<int>& a) {
    vector<int> d;  // d[k] = menor término de subseq. de comprimento k+1
    for (int x : a) {
        auto it = lower_bound(d.begin(), d.end(), x);
        if (it == d.end()) 
            d.push_back(x);
        else 
            *it = x;
    }
    return d;
}

12. Árvores, DP em Árvore e Toposort

O que faz

    Tree DP: otimização em árvores via DFS.

    Toposort: ordenação de vértices em DAG.

Quando usar (Tipos de problema)

    Diâmetro ou profundidade de árvore

    Problemas de cobertura mínima ou caminho em árvore

    Compilação de módulos com dependências

Exemplo de código (C++)

vector<int> order;
vector<bool> vis;
vector<vector<int>> adj;

// DFS para toposort
void dfs(int u) {
    vis[u] = true;
    for (int v : adj[u]) {
        if (!vis[v]) 
            dfs(v);
    }
    order.push_back(u);
}

// Após chamadas a dfs em todos os nós, basta inverter 'order'.

13. DSU (Union-Find) e MST

O que faz

    DSU: gerencia partições dinâmicas com união por rank e compressão de caminho.

    MST (Kruskal): constrói a árvore de custo mínimo.

Quando usar (Tipos de problema)

    Conectividade dinâmica, componentes em grafos

    Construção de redes de custo mínimo

Exemplo de código (C++)

struct DSU {
    vector<int> parent, rank;
    DSU(int n): parent(n, -1), rank(n, 0) {}

    int find(int x) {
        return parent[x] < 0 ? x : parent[x] = find(parent[x]);
    }

    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (rank[a] < rank[b]) swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) rank[a]++;
        return true;
    }
};

14. Segment Tree

O que faz
Suporta consultas (ex.: minimum, sum) e atualizações em intervalos em O(log n), com lazy propagation.

Quando usar (Tipos de problema)

    RMQ/RSQ em arrays estáticos ou dinâmicos

    Atualizações em massa sobre intervalos

Exemplo de código (C++)

struct SegTree {
    int n;
    vector<int> st;
    SegTree(int _n): n(_n), st(4 * _n) {}

    void build(int p, int l, int r, const vector<int>& a) {
        if (l == r) {
            st[p] = a[l];
        } else {
            int m = (l + r) / 2;
            build(p<<1, l, m, a);
            build(p<<1|1, m+1, r, a);
            st[p] = min(st[p<<1], st[p<<1|1]);
        }
    }

    int query(int p, int l, int r, int i, int j) {
        if (r < i || l > j) return INT_MAX;
        if (l >= i && r <= j) return st[p];
        int m = (l + r) / 2;
        return min(
            query(p<<1,     l,   m, i, j),
            query(p<<1|1, m+1, r, i, j)
        );
    }
};

15. Binary Lifting, LCA e Sparse Table

O que faz

    Binary Lifting: pré-processa ancestrais em potências de dois.

    LCA: calcula o ancestral comum mais próximo em O(log n).

    Sparse Table: responde queries idempotentes (min/max/gcd) em O(1) após O(n log n) de pré-processamento.

Quando usar (Tipos de problema)

    Cálculo de distância entre nós em árvores

    Consultas offline sobre intervalos

Exemplo de código (C++)

const int LOG = 20;
vector<int> depth;
int up[N][LOG];

// Preprocessamento via DFS
void dfs(int u, int p) {
    up[u][0] = p;
    for (int j = 1; j < LOG; j++) {
        up[u][j] = up[ up[u][j-1] ][j-1];
    }
    for (int v : adj[u]) {
        if (v != p) {
            depth[v] = depth[u] + 1;
            dfs(v, u);
        }
    }
}

// Calcula LCA de a e b
int lca(int a, int b) {
    if (depth[a] < depth[b]) swap(a, b);
    int diff = depth[a] - depth[b];
    for (int j = 0; j < LOG; j++) {
        if (diff & (1 << j))
            a = up[a][j];
    }
    if (a == b) return a;
    for (int j = LOG - 1; j >= 0; j--) {
        if (up[a][j] != up[b][j]) {
            a = up[a][j];
            b = up[b][j];
        }
    }
    return up[a][0];
}

16. Strings Avançadas (Rolling Hash & KMP)

O que faz

    Rolling Hash: gera hashes de prefixos para comparações de substrings.

    KMP: busca padrão em texto em O(n + m).

Quando usar (Tipos de problema)

    Comparação rápida de substrings (LCS de strings)

    Detecção de padrões, ciclos em strings

Exemplo de código (C++)

using ll = long long;
const ll B = 137, M = 1000000007;
vector<ll> h, p;

// Inicializa hash e potências
void init(const string& s) {
    int n = s.size();
    h.assign(n + 1, 0);
    p.assign(n + 1, 1);
    for (int i = 0; i < n; i++) {
        h[i+1] = (h[i] * B + s[i]) % M;
        p[i+1] = (p[i] * B) % M;
    }
}

// Retorna hash de substring [l, r)
ll getHash(int l, int r) {
    return (h[r] - h[l] * p[r - l] % M + M) % M;
}

// Constrói array LPS para KMP
vector<int> buildLPS(const string& pat) {
    int n = pat.size();
    vector<int> lps(n, 0);
    for (int i = 1, len = 0; i < n; ) {
        if (pat[i] == pat[len]) {
            lps[i++] = ++len;
        } else if (len) {
            len = lps[len - 1];
        } else {
            lps[i++] = 0;
        }
    }
    return lps;
}

// Busca todas as ocorrências de pat em txt
vector<int> kmpSearch(const string& txt, const string& pat) {
    vector<int> lps = buildLPS(pat), res;
    for (int i = 0, j = 0; i < txt.size(); ) {
        if (txt[i] == pat[j]) {
            i++; j++;
        }
        if (j == pat.size()) {
            res.push_back(i - j);
            j = lps[j - 1];
        } else if (i < txt.size() && txt[i] != pat[j]) {
            j ? j = lps[j - 1] : i++;
        }
    }
    return res;
}
