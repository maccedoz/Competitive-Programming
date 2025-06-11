<h1 align="center">BIBlLIOTECA PARA PROGRAMAÇÂO COMPETITIVA</h1>
<h4 align="center"> Por Magno Macedo </h4>

Essa é uma biblioteca em construção, as informações contidas aqui ainda não estão 100% prontas.

# Limites de execução:

- 12 <= O(n!)
- 25 <= O(2^n)
- 100 <= O(n^4)
- 500 <= O(n^3)
- 10^4 <= O(n^2)
- 10^6 <= O(nlogn)
- 10^8 <= O(n)
- 10^8 > O(logn) ou O(1)
  
# Funções gerais do C++

## 1. Manipulação de Strings
- tolower(char c) / toupper(char c): Converte um caractere para minúsculo/maiúsculo.
  
char minuscula = tolower('A'); (resultado: 'a')

- string::find(substring): Retorna a primeira posição de uma substring ou string::npos se não encontrada.
  
string s = "hello world"; size_t pos = s.find("world");

- string::substr(pos, len): Extrai uma parte da string, começando em pos com len caracteres.
  
string s = "programming"; string sub = s.substr(3, 4); (resultado: "gram")

- stoi(string s) / stoll(string s) / stod(string s): Converte string para int, long long ou double.
  
int num = stoi("123");

- to_string(numeric_value): Converte um valor numérico para string.
  
string str_num = to_string(456);

- getline(cin, string s): Lê uma linha inteira do cin (incluindo espaços).
  
string linha; getline(cin, linha);

- reverse(s.begin(), s.end()): Inverte a ordem dos caracteres na string.

string s = "hello"; reverse(s.begin(), s.end()); (resultado: "olleh")

- sort(s.begin(), s.end()): Ordena os caracteres da string lexicograficamente.

string s = "bac"; sort(s.begin(), s.end()); (resultado: "abc")

### Verificação/Conversão de Case (Bitwise): Operações rápidas, sem chamadas de função.

- bool is_digit = (c >= '0' && c <= '9');

- char to_lower_bitwise(char c) { return c | ' '; }

- bool is_lower_bitwise(char c) { return (c & ' ') == ' '; }

## 2. Estruturas de Dados e Algoritmos Padrão (STL - C++)
- sort(begin, end): Ordena um intervalo.

vector<int> v = {5, 2, 8, 1}; sort(v.begin(), v.end()); (resultado: {1, 2, 5, 8})

- min_element(begin, end) / max_element(begin, end): Retorna um iterador para o menor/maior elemento.

int menor = *min_element(v.begin(), v.end());

- accumulate(begin, end, initial_value): Soma elementos de um intervalo.

int soma = accumulate(v.begin(), v.end(), 0);

- count(begin, end, value): Conta ocorrências de um valor em um intervalo.

int ocorrencias = count(v.begin(), v.end(), 5);

- binary_search(begin, end, value): Verifica a existência de um valor em intervalo ordenado.

bool encontrado = binary_search(v.begin(), v.end(), 7);

- lower_bound(begin, end, value) / upper_bound(begin, end, value): Em intervalo ordenado, lower_bound aponta para o 1º elemento ≥ value; upper_bound aponta para o 1º elemento > value.

auto it_lb = lower_bound(v.begin(), v.end(), 5);

- unique(begin, end) + erase: Remove duplicatas consecutivas (após sort).

v.erase(unique(v.begin(), v.end()), v.end());

- next_permutation(begin, end) / prev_permutation(begin, end): Gera a próxima/anterior permutação lexicográfica.

do { cout << s << endl; } while (next_permutation(s.begin(), s.end()));

- iota(begin, end, value): Preenche um intervalo com valores sequenciais.

vector<int> v(5); iota(v.begin(), v.end(), 10); (v será {10, 11, 12, 13, 14})

- nth_element(begin, nth, end): Posiciona o k-ésimo menor elemento na sua posição correta (em O(N) médio).

nth_element(v.begin(), v.begin() + 2, v.end());

- max({a, b, c, ...}) / min({a, b, c, ...}): Retorna o maior/menor de múltiplos valores.

int m = max({5, 8, 3, 12}); (resultado: 12)

##3. Funções Matemáticas
- abs(value) / fabs(value): Retorna o valor absoluto (int / double).

int val_abs = abs(-10);

- pow(base, exponent): Calcula potência (base 
exponent
 ).

double res = pow(2, 3); (resultado: 8.0)

- sqrt(value): Calcula raiz quadrada.

double raiz = sqrt(25.0);

- ceil(value) / floor(value) / round(value): Arredonda para cima, para baixo, ou para o inteiro mais próximo.

ceil(3.14) é 4.0; floor(3.99) é 3.0; round(3.5) é 4.0.

- gcd(a, b) (C++17) / __gcd(a, b) (GCC): Máximo Divisor Comum.

int mdc = gcd(12, 18); (resultado: 6)

- lcm(a, b) (C++17): Mínimo Múltiplo Comum.

int mmc = lcm(4, 6); (resultado: 12)

- partial_sum(begin, end, out_begin): Calcula somas prefixadas.

vector<int> ps; partial_sum(v.begin(), v.end(), back_inserter(ps));

- adjacent_difference(begin, end, out_begin): Calcula diferenças entre elementos adjacentes.

vector<int> ad; adjacent_difference(v.begin(), v.end(), back_inserter(ad));

- Soma/Multiplicação com Wraparound (Módulo): Evita overflow em operações modulares.

long long sum_safe = (0LL + a + b) % MOD;

bool is_mul_overflow(int a, int b) { return b != 0 && a > INT_MAX / b; }

- pow_mod(base, exponent, mod): Exponenciação modular rápida (base 
exponent
 (modmod)).

Função otimizada para long long pow_mod(long long a, long long b, long long mod) { ... }

Teste de Primalidade Miller-Rabin: Verifica se número grande é primo (probabilístico).

##4. Bit Manipulation (Para Otimização)
- __builtin_popcount(x) / __builtin_popcountll(x): Conta bits '1'.

int ones = __builtin_popcount(7); (resultado: 3)

- __builtin_clz(x) / __builtin_clzll(x): Conta zeros à esquerda.

Truque: ⌊log 2(x)⌋ é 31 - __builtin_clz(x) (para unsigned int).

- __builtin_ctz(x): Conta zeros à direita.

int tz = __builtin_ctz(8); (resultado: 3)

- __builtin_parity(x): Retorna 1 se o número de bits '1' for ímpar, 0 se par.

##5. Utilitários Diversos e Truques
- memset(ptr, value, num): Preenche bloco de memória (use com 0 ou -1 para int arrays).

memset(arr, 0, sizeof(arr));

- fill(begin, end, value): Preenche um intervalo com qualquer valor.

fill(v.begin(), v.end(), 7);

- swap(a, b): Troca os valores de duas variáveis.

int x = 5, y = 10; swap(x, y);

- Inicialização vetor multidimensional: vector<vector<int>> dp(n, vector<int>(m, -1));

Custom hash para unordered_map<pair<int, int>>: Permite usar pair como chave em unordered_map.

Exemplo de struct pair_hash { ... } e uso com unordered_map<pair<int,int>, int, pair_hash>.

##6. Debugging e Utilitários de Teste

- Medição de Tempo (<chrono>): Mede tempo de execução.

auto start = chrono::high_resolution_clock::now(); ... auto end = chrono::high_resolution_clock::now();

##Dicas Adicionais

- Otimização I/O: ios_base::sync_with_stdio(false); cin.tie(NULL); no main().


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
// Número de elementos no conjunto original (0, 1, 2, 3)

for (int mask = 0; mask < (1 << n); ++mask) {
    // Percorre todas as possíveis máscaras de bits de tamanho n
    // (1 << n) é 2^n — o número total de subconjuntos possíveis

    vector<int> subset;
    // Vetor para armazenar o subconjunto correspondente à máscara atual

    for (int i = 0; i < n; ++i) {
        // Para cada bit da máscara (de 0 a n-1)

        if (mask & (1 << i))
            // Verifica se o i-ésimo bit está ligado (1) na máscara atual
            // Se estiver, significa que o elemento i faz parte do subconjunto

            subset.push_back(i);
            // Adiciona o elemento i ao subconjunto
    }

    // Aqui você poderia, por exemplo, imprimir ou processar o subset
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
// Função que resolve o problema da Mochila 0/1
// W é a capacidade máxima da mochila
// w é o vetor com os pesos dos itens
// v é o vetor com os valores dos itens
int knap(int W, const vector<int>& w, const vector<int>& v) {
    int n = w.size(); // Número de itens

    // Cria a matriz dp com (n+1) linhas e (W+1) colunas, inicializada com zero
    // dp[i][j] representa o maior valor possível usando os primeiros i itens com capacidade j
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    // Preenche a matriz dp
    for (int i = 1; i <= n; ++i) { // Para cada item (1 até n)
        for (int j = 0; j <= W; ++j) { // Para cada capacidade de mochila (0 até W)
            
            // Caso não pegue o item i-1
            int sem_item = dp[i - 1][j];

            // Caso pegue o item i-1 (só se couber na mochila)
            int com_item = 0;
            if (j >= w[i - 1]) {
                // Soma o valor do item i-1 com o melhor valor para a capacidade restante
                com_item = dp[i - 1][j - w[i - 1]] + v[i - 1];
            }

            // Armazena o melhor valor entre pegar ou não o item
            dp[i][j] = max(sem_item, com_item);
        }
    }

    // Retorna o melhor valor possível com todos os itens e capacidade total W
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

## Algoritmos Úteis de Teoria dos Números

### 1. Máximo Divisor Comum (GCD) - Algoritmo de Euclides

```cpp
int gcd(int a, int b) {
    return b ? gcd(b, a % b) : a;
}
```
**Uso:** Calcula o maior número que divide `a` e `b` sem deixar resto. Muito usado para simplificar frações, calcular o mínimo múltiplo comum (MMC), e resolver problemas de divisibilidade.

---

### 2. Exponenciação Modular Rápida

```cpp
int modPow(int base, int exp, int m) {
    long long result = 1, b = base;
    while (exp > 0) {
        if (exp & 1) result = (result * b) % m;
        b = (b * b) % m;
        exp >>= 1;
    }
    return result;
}
```
**Uso:** Calcula `(base^exp) % m` de forma eficiente em O(log exp). Fundamental em criptografia (RSA), hashing e aritmética modular em geral.

---

### 3. Inverso Modular (Fermat)

```cpp
int modinv(int a, int m) {
    return modPow(a, m - 2, m);
}
```
**Uso:** Calcula o inverso de `a` módulo `m`, ou seja, o número `x` tal que `a * x ≡ 1 (mod m)`. Só funciona se `m` é primo (baseado no pequeno teorema de Fermat).

---

### 4. Algoritmo Estendido de Euclides

```cpp
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
**Uso:** Encontra inteiros `x` e `y` que satisfazem `ax + by = gcd(a, b)`. Útil para resolver equações diofantinas e obter inversos modulares quando `m` não é primo.

---

### 5. Crivo de Eratóstenes

```cpp
vector<bool> is_prime(N+1, true);
is_prime[0] = is_prime[1] = false;
for (int i = 2; i * i <= N; i++) {
    if (is_prime[i]) {
        for (int j = i * i; j <= N; j += i)
            is_prime[j] = false;
    }
}
```
**Uso:** Pré-processa todos os números primos até `N`. Rápido e eficiente (O(N log log N)). Base para problemas de contagem e fatoração.

---

### 6. Fatoração de um número

```cpp
vector<int> fatorar(int n) {
    vector<int> fatores;
    for (int i = 2; i * i <= n; i++) {
        while (n % i == 0) {
            fatores.push_back(i);
            n /= i;
        }
    }
    if (n > 1) fatores.push_back(n);
    return fatores;
}
```
**Uso:** Decompõe um número em seus fatores primos. Útil em problemas que exigem análise da estrutura multiplicativa de um número.

---

### 7. Função Totiente de Euler (φ)

```cpp
int phi(int n) {
    int res = n;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0) n /= i;
            res -= res / i;
        }
    }
    if (n > 1) res -= res / n;
    return res;
}
```
**Uso:** Conta quantos inteiros positivos menores que `n` são coprimos com `n`. Essencial em problemas envolvendo ciclos, ordens e exponenciação modular.

---

### 8. Sieve de φ(n)

```cpp
vector<int> phi(N+1);
for (int i = 0; i <= N; i++) phi[i] = i;
for (int i = 2; i <= N; i++) {
    if (phi[i] == i) {
        for (int j = i; j <= N; j += i)
            phi[j] -= phi[j] / i;
    }
}
```
**Uso:** Pré-calcula o valor de φ(n) para todos `n` até `N`. Muito eficiente para resolver múltiplas queries relacionadas a coprimos.

---

### 9. Contagem e Soma dos Divisores

```cpp
pair<int, int> contarSomaDivisores(int n) {
    int qtd = 1, soma = 1;
    for (int i = 2; i * i <= n; i++) {
        int pot = 0, powsoma = 1, base = 1;
        while (n % i == 0) {
            pot++;
            n /= i;
            base *= i;
            powsoma += base;
        }
        qtd *= (pot + 1);
        soma *= powsoma;
    }
    if (n > 1) {
        qtd *= 2;
        soma *= (1 + n);
    }
    return {qtd, soma};
}
```
**Uso:** Retorna o número total de divisores e a soma dos divisores de `n`. Útil em teoria multiplicativa e problemas de otimização.

---

### 10. Resolver ax ≡ b (mod m)

```cpp
int solve_congruence(int a, int b, int m) {
    int x, y;
    int g = extgcd(a, m, x, y);
    if (b % g != 0) return -1;
    x = (x * (b / g)) % m;
    return (x + m) % m;
}
```
**Uso:** Resolve congruência linear. Retorna uma solução `x` de `ax ≡ b (mod m)`, se existir. Usa o algoritmo estendido de Euclides.

---

### 11. Teorema Chinês do Resto (CRT) para 2 módulos

```cpp
long long CRT(long long a1, long long m1, long long a2, long long m2) {
    long long x1, y1;
    long long g = extgcd(m1, m2, x1, y1);
    if ((a2 - a1) % g != 0) return -1;
    long long mod = m1 * m2;
    long long res = (a1 + m1 * ((x1 * ((a2 - a1) / g)) % m2)) % mod;
    return (res + mod) % mod;
}
```
**Uso:** Resolve sistemas de duas congruências com módulos coprimos. Muito útil para reconstrução de restos e aplicações em criptografia.

---

Esses algoritmos formam uma **base sólida de teoria dos números** para programação competitiva e resolução de problemas matemáticos avançados.

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
## 17. Rolling Hash

O que faz:
Rolling Hash (ou hash deslizante) é uma técnica para comparar substrings de forma rápida, transformando uma string (ou substring) em um número inteiro (hash). Isso permite verificar se duas substrings são iguais sem precisar comparar caractere por caractere.


Quando usar:
Comparação rápida de substrings, LCS de strings, detecção de colisões.

Código C++:
```cpp
using ll = long long;
const ll B = 137, M = 1000000007;
vector<ll> h, p;

// Inicializa os vetores de hash e potências
void init(const string& s) {
    int n = s.size();
    h.assign(n + 1, 0);
    p.assign(n + 1, 1);
    for (int i = 0; i < n; i++) {
        h[i+1] = (h[i] * B + s[i]) % M;
        p[i+1] = (p[i] * B) % M;
    }
}

// Retorna o hash da substring [l, r)
ll getHash(int l, int r) {
    return (h[r] - h[l] * p[r - l] % M + M) % M;
}
```
## 18. KMP (Knuth-Morris-Pratt)

O que faz:
O KMP é um algoritmo eficiente para buscar todas as ocorrências de um padrão (string pat) dentro de um texto maior (string txt). Ele faz isso em tempo O(n + m).

Quando usar:
Contar ocorrências, encontrar padrões, detectar ciclos em strings.

Código C++:
```cpp
// Constrói array LPS (Longest Prefix Suffix)
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

// Busca todas as ocorrências de 'pat' em 'txt'
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
