# Técnicas de Programação
## 1. Lower/Upper Bound e Busca Binária
**O que faz:**
Busca binária (e funções `lower_bound`/`upper_bound`) localiza posições em um array ordenado
o O(log n). O `upper_bound` retorna um ponteiro no elemento maior ou igual a `x`, enquanto o `upper_bound` retorna o estritamente maior.

**Quando usar (Tipo de problema):**
- Busca de valor específico: encontrar se um elemento existe.
- Busca de primeiro elemento ≥/≤ valor: determinar limites inferior e superior.
- Decisões monotônicas: otimização por teste de satisfabilidade em pesquisas paramétricas.
**Exemplo de código (C++):**
```cpp
#include <bits/stdc++.h>
using namespace std;

// Função que verifica se é possível reduzir todos os fragmentos maiores que 'valor'
// usando no máximo 'maxOperacoes' operações, considerando os fragmentos até o meio da lista
bool ok(int valor, vector<int> listaFragmentos, int maxOperacoes, int tamanho) {
    int operacoes = 0;
    int meio = tamanho / 2;

    // Itera da metade para o início, verificando fragmentos maiores que 'valor'
    for (int i = meio; i >= 0 && listaFragmentos[i] > valor; i--) {
        // Calcula quantas operações são necessárias para reduzir o fragmento ao valor desejado
        operacoes += listaFragmentos[i] - valor;

        // Se exceder o limite máximo de operações, retorna falso imediatamente
        if (operacoes > maxOperacoes) {
            return false;
        }
    }
    // Se não ultrapassar o limite, é possível com 'valor' e retorna verdadeiro
    return true;
}

int main() {
    int quantidadeFragmentos, maxOperacoes;
    cin >> quantidadeFragmentos >> maxOperacoes;

    vector<int> listaFragmentos(quantidadeFragmentos);
    for (int i = 0; i < quantidadeFragmentos; i++) {
        cin >> listaFragmentos[i];
    }

    sort(listaFragmentos.begin(), listaFragmentos.end());

    int max = listaFragmentos[quantidadeFragmentos / 2];
    int min = listaFragmentos[quantidadeFragmentos / 2] - maxOperacoes;
    int best = max;  // Guarda a melhor solução encontrada

    while (max > min) {
        int meio = min + (max - min) / 2;
        if (ok(meio, listaFragmentos, maxOperacoes, quantidadeFragmentos)) {
            best = meio;
            max = meio;
        }
        else {
            min = meio + 1;
        }
    }
    cout << best << endl;
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
  int sum = 0;       // Soma atual da janela (subarray)
  int l = 0;         // Índice do início da janela (esquerda)
  int best = 0;      // Melhor tamanho encontrado até agora

  // Percorre o vetor com o índice da direita da janela
  for (int r = 0; r < a.size(); ++r) {
    sum += a[r];     // Adiciona o elemento da direita na soma da janela

    // Ajusta a janela para garantir que a soma ≤ K
    while (sum > K) {
      sum -= a[l];   // Remove o elemento mais à esquerda da janela
      l++;           // Move o início da janela para a direita
    }

    // Atualiza o maior tamanho válido encontrado
    best = max(best, r - l + 1);
  }
  
  return best;       // Retorna o tamanho máximo da subarray com soma ≤ K
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
// Retorna todos os pares (a[l], a[r]) cuja soma é igual a X
vector<pair<int,int>> pairsWithSum(const vector<int>& a, int X) {
    vector<pair<int,int>> res;       // vetor para armazenar os pares encontrados
    int l = 0;                       // ponteiro esquerdo no início do vetor
    int r = (int)a.size() - 1;       // ponteiro direito no final do vetor

    while (l < r) {                  // enquanto os ponteiros não se cruzarem
        int s = a[l] + a[r];         // soma dos elementos apontados

        if (s == X) {                // se a soma é igual a X
            res.emplace_back(a[l], a[r]); // adiciona o par ao resultado
            l++;                    // move ponteiro esquerdo para frente
            r--;                    // move ponteiro direito para trás
        }
        else if (s < X)              // se a soma é menor que X
            l++;                    // aumenta soma movendo ponteiro esquerdo para frente
        else                        // se a soma é maior que X
            r--;                    // diminui soma movendo ponteiro direito para trás
    }

    return res;                     // retorna todos os pares encontrados
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

# Grafos e Caminhos Mínimos

---

## 10. Representação de Grafos

| Tipo                 | Quando Usar                                  | Vantagens                       | Desvantagens                  |
|----------------------|---------------------------------------------|--------------------------------|------------------------------|
| **Lista de Adjacência** | Grafos esparsos (arestas ~ vértices)       | Memória O(V + E), fácil iteração| Busca de aresta O(V)          |
| **Matriz de Adjacência**| Grafos densos ou pequenos (ex: grafos completos) | Busca de aresta O(1), simples | Memória O(V²), lenta em grafos esparsos |

**Exemplo (C++):**

```cpp
// Lista de adjacência
int n;
vector<vector<int>> g(n);
g[u].push_back(v); // adiciona aresta u->v

// Matriz de adjacência
vector<vector<int>> M(n, vector<int>(n, 0));
M[u][v] = 1;       // define aresta u->v
```

11. DFS (Busca em Profundidade)

Usos Comuns:

    Encontrar componentes conexas

    Detectar ciclos

    Ordenação topológica

Pré-requisitos: vetor visited, recursão ou pilha.

Código (C++):
```cpp
void dfs(int u) {
    vis[u] = true;
    for (int v : g[u])
        if (!vis[v])
            dfs(v);
}

int componentes = 0;
for (int i = 0; i < n; i++) {
    if (!vis[i]) {
        dfs(i);
        componentes++;
    }
}

```
Detecção de ciclo em grafo dirigido:

``` cpp
enum Estado { BRANCO, CINZA, PRETO };
vector<Estado> estado(n, BRANCO);
bool ciclo = false;

void dfsCiclo(int u) {
    estado[u] = CINZA;
    for (int v : g[u]) {
        if (estado[v] == CINZA) ciclo = true;
        else if (estado[v] == BRANCO) dfsCiclo(v);
    }
    estado[u] = PRETO;
}
```

8. BFS (Busca em Largura)

Usos Comuns:

    Caminho mínimo em grafo não ponderado

    Verificar distância entre vértices

    Testar bipartição do grafo

Pré-requisitos: fila, vetor de distâncias, vetor de predecessores.

Código (C++):
``` cpp
vector<int> dist(n, -1);
queue<int> q;

dist[src] = 0;
q.push(src);

while (!q.empty()) {
    int u = q.front(); q.pop();
    for (int v : g[u]) {
        if (dist[v] == -1) {
            dist[v] = dist[u] + 1;
            q.push(v);
        }
    }
}
```
Teste de bipartição:

    Colorir vértices em duas cores alternadamente durante BFS.

    Se encontrar aresta ligando vértices da mesma cor, grafo não é bipartido.

9. Dijkstra

Para que serve:
Caminho mínimo em grafos com pesos positivos.

Aplicações:

    GPS e roteamento

    Redes de estradas

Pré-requisitos:
Lista de adjacência com pesos, heap de prioridade.

Código (C++):
``` cpp
using P = pair<long long,int>;
vector<long long> dist(n, LLONG_MAX);
priority_queue<P, vector<P>, greater<P>> pq;

dist[src] = 0;
pq.emplace(0, src);

while (!pq.empty()) {
    auto [d, u] = pq.top(); pq.pop();
    if (d != dist[u]) continue;
    for (auto [v, w] : g[u]) {
        if (dist[u] + w < dist[v]) {
            dist[v] = dist[u] + w;
            pq.emplace(dist[v], v);
        }
    }
}
```
10. Bellman-Ford

Para que serve:
Caminho mínimo com pesos negativos e detecção de ciclos negativos.

Aplicações:

    Mercado financeiro (arbitragem)

    Sistemas com restrições temporais negativas

Pré-requisitos:
Lista de arestas, vetor de distâncias.

Código (C++):
``` cpp
const long long INF = 4e18;
vector<long long> dist(n, INF);
dist[src] = 0;

for (int i = 1; i < n; i++) {
    for (auto [u, v, w] : edges) {
        if (dist[u] + w < dist[v])
            dist[v] = dist[u] + w;
    }
}

// Detecção de ciclo negativo
bool cicloNegativo = false;
for (auto [u, v, w] : edges) {
    if (dist[u] + w < dist[v])
        cicloNegativo = true;
}

```
---

# Módulo Avançado B

## 11. Teoria dos Números

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
```
## 12. LIS e DP Bitmask

O que faz

    LIS: encontra a maior subsequência estritamente crescente em O(n log n).

    DP Bitmask: resolve problemas sobre todos os subconjuntos (n ≤ 20–25) usando bits como estados.

Quando usar (Tipos de problema)

    LIS: padrões de crescimento em sequências (ex.: empilhamento de caixas).

    DP Bitmask: TSP, cobertura de conjuntos, atribuição de tarefas.

Exemplo de código (C++)
``` cpp
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
```

## 13. Árvores, DP em Árvore e Toposort

O que faz

    Tree DP: otimização em árvores via DFS.

    Toposort: ordenação de vértices em DAG.

Quando usar (Tipos de problema)

    Diâmetro ou profundidade de árvore

    Problemas de cobertura mínima ou caminho em árvore

    Compilação de módulos com dependências

Exemplo de código (C++)
``` cpp
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
```
## 14. DSU (Union-Find) e MST

O que faz

    DSU: gerencia partições dinâmicas com união por rank e compressão de caminho.

    MST (Kruskal): constrói a árvore de custo mínimo.

Quando usar (Tipos de problema)

    Conectividade dinâmica, componentes em grafos

    Construção de redes de custo mínimo

Exemplo de código (C++)
``` cpp
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
```
## 15. Segment Tree

O que faz
Suporta consultas (ex.: minimum, sum) e atualizações em intervalos em O(log n), com lazy propagation.

Quando usar (Tipos de problema)

    RMQ/RSQ em arrays estáticos ou dinâmicos

    Atualizações em massa sobre intervalos

Exemplo de código (C++)
``` cpp
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
``` 
## 16. Binary Lifting, LCA e Sparse Table

O que faz

    Binary Lifting: pré-processa ancestrais em potências de dois.

    LCA: calcula o ancestral comum mais próximo em O(log n).

    Sparse Table: responde queries idempotentes (min/max/gcd) em O(1) após O(n log n) de pré-processamento.

Quando usar (Tipos de problema)

    Cálculo de distância entre nós em árvores

    Consultas offline sobre intervalos

Exemplo de código (C++)
``` cpp
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
```
## 17. Strings Avançadas (Rolling Hash & KMP)

O que faz

    Rolling Hash: gera hashes de prefixos para comparações de substrings.

    KMP: busca padrão em texto em O(n + m).

Quando usar (Tipos de problema)

    Comparação rápida de substrings (LCS de strings)

    Detecção de padrões, ciclos em strings

Exemplo de código (C++)
``` cpp
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
``` 
