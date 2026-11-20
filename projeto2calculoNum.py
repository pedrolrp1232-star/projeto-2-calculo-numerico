#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 CÁLCULO NUMÉRICO – ATIVIDADE COMPLEMENTAR (LES710C)
===============================================================================
 Questões resolvidas (uma seção para cada uma):

   SEÇÃO 1 – Questão 01.1  Sistemas lineares – métodos diretos
                           (Eliminação de Gauss com pivotamento parcial + LU)
   SEÇÃO 2 – Questão 02.1  Sistemas lineares – métodos iterativos
                           (Ponte de Wheatstone resolvida por Gauss-Seidel)
   SEÇÃO 3 – Questão 03.2  Interpolação polinomial
                           (formas de Lagrange e de Newton, graus 2 a 4)
   SEÇÃO 4 – Questão 04.3  Integração numérica
                           (Regras do Trapézio e de Simpson repetidas)

 Como usar:
   python atividade_calculo_numerico.py          -> menu interativo
   python atividade_calculo_numerico.py --auto   -> roda as 4 questões com os
                                                    dados do enunciado, sem perguntas
   python atividade_calculo_numerico.py --graficos  -> menu com gráficos ligados

 Em todas as perguntas, apenas ENTER mantém o valor padrão (dados do enunciado).
 Números podem ser digitados com ponto ou vírgula (2.5 ou 2,5).
 Dependências: apenas a biblioteca padrão. O numpy (verificações extras) e o
 matplotlib (gráficos) são opcionais.
===============================================================================
"""

import itertools
import math
import os
import sys

try:
    import numpy as np
except ImportError:          # o programa funciona sem numpy (só perde as verificações extras)
    np = None

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

AUTO = "--auto" in sys.argv                # modo sem perguntas (usa dados do enunciado)
GRAFICOS = "--graficos" in sys.argv        # liga/desliga gráficos (também no menu)
LARGURA = 78


# =============================================================================
#  UTILIDADES: impressão e leitura de dados
# =============================================================================
def linha(c="-"):
    print(c * LARGURA)


def titulo(txt):
    print()
    linha("=")
    print(txt)
    linha("=")


def subtitulo(txt):
    print()
    print(txt)
    linha("-")


def passo(n, txt):
    subtitulo(f"PASSO {n} — {txt}")


def _entrada(msg):
    try:
        return input(msg)
    except EOFError:
        print()
        raise SystemExit(0)


def _num(txt):
    return float(txt.strip().replace(",", "."))


def ler_float(msg, padrao=None):
    if AUTO and padrao is not None:
        return padrao
    suf = f" [{padrao:g}]" if padrao is not None else ""
    while True:
        s = _entrada(f"  {msg}{suf}: ").strip()
        if s == "" and padrao is not None:
            return padrao
        try:
            return _num(s)
        except ValueError:
            print("   ! valor inválido, digite um número (ex.: 2.5 ou 2,5).")


def ler_int(msg, padrao, minimo=1, maximo=None):
    if AUTO:
        return padrao
    while True:
        s = _entrada(f"  {msg} [{padrao}]: ").strip()
        if s == "":
            return padrao
        try:
            v = int(s)
        except ValueError:
            print("   ! digite um número inteiro.")
            continue
        if v < minimo or (maximo is not None and v > maximo):
            lim = f"entre {minimo} e {maximo}" if maximo is not None else f">= {minimo}"
            print(f"   ! o valor deve ser {lim}.")
            continue
        return v


def ler_lista(msg, qtd=None, padrao=None):
    """Lê uma lista de números separados por espaço (ou ';')."""
    if AUTO and padrao is not None:
        return list(padrao)
    suf = f" [{' '.join(f'{v:g}' for v in padrao)}]" if padrao is not None else ""
    while True:
        s = _entrada(f"  {msg}{suf}: ").strip()
        if s == "" and padrao is not None:
            return list(padrao)
        try:
            vals = [_num(t) for t in s.replace(";", " ").split()]
        except ValueError:
            print("   ! há valores inválidos; separe os números por espaço.")
            continue
        if qtd is not None and len(vals) != qtd:
            print(f"   ! informe exatamente {qtd} valores (você digitou {len(vals)}).")
            continue
        if not vals:
            print("   ! lista vazia.")
            continue
        return vals


def ler_nomes(msg, qtd, prefixo):
    padrao = [f"{prefixo} {i + 1}" for i in range(qtd)]
    if AUTO:
        return padrao
    s = _entrada(f"  {msg} (separados por ';'; ENTER = {prefixo} 1..{qtd}): ").strip()
    if not s:
        return padrao
    nomes = [t.strip() for t in s.split(";") if t.strip()]
    if len(nomes) != qtd:
        print("   ! quantidade diferente do esperado; usando nomes genéricos.")
        return padrao
    return nomes


def sim_nao(msg, padrao=True):
    if AUTO:
        return padrao
    dica = "S/n" if padrao else "s/N"
    while True:
        s = _entrada(f"  {msg} [{dica}]: ").strip().lower()
        if s == "":
            return padrao
        if s in ("s", "sim", "y", "yes"):
            return True
        if s in ("n", "nao", "não", "no"):
            return False
        print("   ! responda com s ou n.")


def eq_texto(coefs, nomes, rhs):
    """Monta o texto de uma equação linear, omitindo coeficientes nulos."""
    partes = []
    for c, nm in zip(coefs, nomes):
        if abs(c) < 1e-15:
            continue
        mod = abs(c)
        termo = nm if abs(mod - 1) < 1e-15 else f"{mod:g}·{nm}"
        partes.append(("-" if c < 0 else "+", termo))
    if not partes:
        return f"0 = {rhs:g}"
    s = ("-" if partes[0][0] == "-" else "") + partes[0][1]
    for sg, t in partes[1:]:
        s += f" {sg} {t}"
    return f"{s} = {rhs:g}"


def mostrar_sistema(A, b, nomes, rotulos=None, casas=4):
    """Imprime a matriz aumentada [A | b] com os nomes das incógnitas no topo."""
    n = len(A)
    larg = max(9, casas + 6)
    pref = " " * (14 if rotulos else 3)
    print(pref + " ".join(f"{nm:>{larg}}" for nm in nomes) + "  |  " + f"{'b':>{larg}}")
    for i in range(n):
        r = f"{rotulos[i]:<14}" if rotulos else "   "
        print(r + " ".join(f"{A[i][j]:{larg}.{casas}f}" for j in range(len(A[i]))) +
              "  |  " + f"{b[i]:{larg}.{casas}f}")


def mostrar_matriz(M, nome, casas=4):
    print(f"   {nome} =")
    for r in M:
        print("     [ " + "  ".join(f"{v:10.{casas}f}" for v in r) + " ]")


def residuo_max(A, x, b):
    return max(abs(sum(A[i][j] * x[j] for j in range(len(x))) - b[i]) for i in range(len(b)))


def _plt():
    """Importa o matplotlib só quando um gráfico é pedido."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  (matplotlib não está instalado: gráfico ignorado)")
        return None


def salvar_e_mostrar(plt, fig, nome_arquivo):
    pasta = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(pasta, nome_arquivo)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"  Gráfico salvo em: {caminho}")
    if not AUTO:
        plt.show()
    plt.close(fig)


# =============================================================================
#  SEÇÃO 1 – QUESTÃO 01.1 – SISTEMAS LINEARES: MÉTODOS DIRETOS
# =============================================================================
def gauss_pivotamento(A, b, detalhar=True):
    """Eliminação de Gauss com pivotamento parcial + retrossubstituição."""
    n = len(A)
    M = [[float(v) for v in A[i]] + [float(b[i])] for i in range(n)]
    tol = 1e-12 * max(abs(v) for r in A for v in r)

    if detalhar:
        print("  Matriz aumentada inicial [A | b]:")
        mostrar_sistema([r[:n] for r in M], [r[n] for r in M], [f"x{j + 1}" for j in range(n)])

    for k in range(n - 1):
        if detalhar:
            print(f"\n  ► Etapa {k + 1}: zerar os elementos abaixo do pivô, coluna {k + 1}")
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < tol:
            raise ValueError("Sistema singular (pivô nulo): não existe solução única.")
        if p != k:
            M[k], M[p] = M[p], M[k]
            if detalhar:
                print(f"    Pivotamento parcial: troca L{k + 1} ↔ L{p + 1} "
                      f"(maior |elemento| da coluna = {abs(M[k][k]):.4f})")
        elif detalhar:
            print(f"    Pivô: a{k + 1}{k + 1} = {M[k][k]:.4f} (já é o maior da coluna)")
        for i in range(k + 1, n):
            m = M[i][k] / M[k][k]
            if detalhar:
                print(f"    m{i + 1}{k + 1} = {M[i][k]:.4f}/{M[k][k]:.4f} = {m:.6f}"
                      f"   →   L{i + 1} ← L{i + 1} − ({m:.6f})·L{k + 1}")
            for j in range(k, n + 1):
                M[i][j] -= m * M[k][j]
            M[i][k] = 0.0
        if detalhar:
            mostrar_sistema([r[:n] for r in M], [r[n] for r in M], [f"x{j + 1}" for j in range(n)])

    if abs(M[n - 1][n - 1]) < tol:
        raise ValueError("Sistema singular (pivô nulo): não existe solução única.")

    if detalhar:
        print("\n  ► Retrossubstituição (de baixo para cima):")
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
        if detalhar:
            print(f"    x{i + 1} = (b{i + 1} − Σ a·x) / a{i + 1}{i + 1} = "
                  f"({M[i][n]:.4f} − {s:.4f}) / {M[i][i]:.4f} = {x[i]:.6f}")
    return x


def decomposicao_lu(A):
    """Fatoração PA = LU (Doolittle) com pivotamento parcial."""
    n = len(A)
    U = [[float(v) for v in r] for r in A]
    L = [[0.0] * n for _ in range(n)]
    perm = list(range(n))
    tol = 1e-12 * max(abs(v) for r in A for v in r)
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(U[i][k]))
        if abs(U[p][k]) < tol:
            raise ValueError("Sistema singular (pivô nulo): não existe solução única.")
        if p != k:
            U[k], U[p] = U[p], U[k]
            perm[k], perm[p] = perm[p], perm[k]
            L[k][:k], L[p][:k] = L[p][:k], L[k][:k]
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]
            U[i][k] = 0.0
    for i in range(n):
        L[i][i] = 1.0
    return L, U, perm


def resolver_lu(L, U, perm, b):
    n = len(b)
    pb = [b[perm[i]] for i in range(n)]
    y = [0.0] * n
    for i in range(n):                                   # L·y = P·b
        y[i] = pb[i] - sum(L[i][j] * y[j] for j in range(i))
    x = [0.0] * n
    for i in range(n - 1, -1, -1):                       # U·x = y
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return pb, y, x


def dados_producao():
    """Dados do problema 01.1 (ou novos dados digitados pelo usuário)."""
    comp_pad = ["Transistor", "Resistor", "Chip de computador"]
    mat_pad = ["Cobre", "Zinco", "Vidro"]
    consumo_pad = [[4, 1, 2],      # transistor: cobre, zinco, vidro
                   [3, 3, 1],      # resistor
                   [2, 1, 3]]      # chip
    disp_pad = [960, 510, 610]
    if sim_nao("Usar os dados do enunciado (transistor/resistor/chip)?", True):
        return comp_pad, mat_pad, consumo_pad, disp_pad
    n = ler_int("Quantos tipos de componentes (= quantos materiais)", 3, 1, 12)
    comp = ler_nomes("Nomes dos componentes", n, "Componente")
    mat = ler_nomes("Nomes dos materiais", n, "Material")
    consumo = []
    print(f"\n  Informe quantas unidades de cada material ({'; '.join(mat)}) "
          f"cada componente usa:")
    for i in range(n):
        consumo.append(ler_lista(f"Consumo de 1 '{comp[i]}'", qtd=n))
    disp = ler_lista(f"Quantidade disponível de cada material ({'; '.join(mat)})", qtd=n)
    return comp, mat, consumo, disp


def resolver_producao(comp, mat, consumo, disp):
    n = len(comp)
    xs = [f"x{j + 1}" for j in range(n)]

    passo(1, "Modelagem do problema como sistema de equações lineares")
    print("  Variáveis (quantidade produzida na semana):")
    for j in range(n):
        print(f"     {xs[j]} = nº de unidades de '{comp[j]}'")
    print("\n  Balanço de cada material (consumo total = quantidade disponível):")
    A = [[consumo[j][i] for j in range(n)] for i in range(n)]   # linhas = materiais
    b = [float(v) for v in disp]
    for i in range(n):
        print(f"     {mat[i]:<10}: {eq_texto(A[i], xs, b[i])}")
    print("\n  Forma matricial A·x = b  (A = consumo transposto: linha = material):")
    mostrar_sistema(A, b, xs, rotulos=mat)

    passo(2, "Método direto 1: Eliminação de Gauss com pivotamento parcial")
    try:
        x_gauss = gauss_pivotamento(A, b)
    except ValueError as e:
        print(f"\n  ✗ {e}")
        return

    passo(3, "Método direto 2: Decomposição LU (PA = LU) — conferência")
    L, U, perm = decomposicao_lu(A)
    print(f"  Permutação das linhas: P·b usa as linhas {[p + 1 for p in perm]} de b")
    mostrar_matriz(L, "L")
    mostrar_matriz(U, "U")
    pb, y, x_lu = resolver_lu(L, U, perm, b)
    PA = [A[perm[i]] for i in range(n)]
    LU = [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    erro_fat = max(abs(PA[i][j] - LU[i][j]) for i in range(n) for j in range(n))
    print(f"   Conferência da fatoração: máx|PA − LU| = {erro_fat:.2e}")
    print("   Substituição direta  L·y = P·b :  y = [" + ", ".join(f"{v:.6f}" for v in y) + "]")
    print("   Substituição reversa U·x = y   :  x = [" + ", ".join(f"{v:.6f}" for v in x_lu) + "]")

    passo(4, "Verificação")
    print(f"  Resíduo Gauss : máx|A·x − b| = {residuo_max(A, x_gauss, b):.2e}")
    print(f"  Resíduo LU    : máx|A·x − b| = {residuo_max(A, x_lu, b):.2e}")
    print(f"  Diferença entre os dois métodos: "
          f"{max(abs(a - c) for a, c in zip(x_gauss, x_lu)):.2e}")
    if np is not None:
        x_np = np.linalg.solve(np.array(A, float), np.array(b, float))
        print(f"  Diferença para numpy.linalg.solve: "
              f"{max(abs(a - c) for a, c in zip(x_gauss, x_np)):.2e}")
    print("\n  Prova real (consumo de cada material com a solução encontrada):")
    for i in range(n):
        termos = " + ".join(f"{A[i][j]:g}×{x_gauss[j]:.4g}" for j in range(n))
        print(f"     {mat[i]:<10}: {termos} = {sum(A[i][j] * x_gauss[j] for j in range(n)):.4f}"
              f"   (disponível: {b[i]:g})")

    subtitulo("RESPOSTA — meta de produção da semana")
    for j in range(n):
        print(f"   {comp[j]:<22}: {x_gauss[j]:.4f}")
    if any(v < -1e-9 for v in x_gauss):
        print("\n   ⚠ Há quantidade negativa: com esses materiais a meta é inviável.")
    elif any(abs(v - round(v)) > 1e-6 for v in x_gauss):
        print("\n   ⚠ A solução não é inteira: na prática seria preciso arredondar "
              "(e o consumo não bateria exatamente).")
    else:
        print("\n   ✓ Solução inteira e não negativa: a meta é realizável usando todo o material.")


def questao_01():
    titulo("SEÇÃO 1 — QUESTÃO 01.1 — SISTEMAS LINEARES (MÉTODOS DIRETOS)")
    print("  Enunciado: uma companhia fabrica transistores, resistores e chips usando cobre,")
    print("  zinco e vidro. Em uma semana há 960 un. de cobre, 510 de zinco e 610 de vidro.")
    print("  Montar o sistema de equações e determinar quantos de cada componente produzir.")
    while True:
        comp, mat, consumo, disp = dados_producao()
        resolver_producao(comp, mat, consumo, disp)
        if not sim_nao("\nFazer um novo cálculo nesta questão?", False):
            break


# =============================================================================
#  SEÇÃO 2 – QUESTÃO 02.1 – MÉTODOS ITERATIVOS: GAUSS-SEIDEL (PONTE DE WHEATSTONE)
# =============================================================================
def raio_espectral_gs(A):
    """ρ da matriz de iteração de Gauss-Seidel: T = −(D+L)⁻¹·U  (precisa de numpy)."""
    if np is None:
        return None
    M = np.array(A, float)
    if np.any(np.abs(np.diag(M)) < 1e-12):
        return math.inf
    T = -np.linalg.solve(np.tril(M), np.triu(M, 1))
    return float(max(abs(np.linalg.eigvals(T))))


def analisar_convergencia(A, mostrar=True):
    """Critério das linhas, critério de Sassenfeld e raio espectral."""
    n = len(A)
    diag_ok = all(abs(A[i][i]) > 1e-12 for i in range(n))
    if not diag_ok:
        if mostrar:
            print("  ✗ Há elemento nulo na diagonal principal: Gauss-Seidel não pode ser aplicado.")
        return {"ok": False, "rho": math.inf}
    # (a) critério das linhas (diagonal estritamente dominante)
    dom = all(abs(A[i][i]) > sum(abs(A[i][j]) for j in range(n) if j != i) for i in range(n))
    # (b) critério de Sassenfeld
    beta = []
    for i in range(n):
        s = sum(abs(A[i][j]) * beta[j] for j in range(i)) + \
            sum(abs(A[i][j]) for j in range(i + 1, n))
        beta.append(s / abs(A[i][i]))
    sass = max(beta) < 1
    # (c) condição necessária e suficiente: ρ(T_GS) < 1
    rho = raio_espectral_gs(A)
    if mostrar:
        print(f"  (a) Critério das linhas (diagonal dominante): {'satisfeito' if dom else 'NÃO satisfeito'}")
        print(f"  (b) Critério de Sassenfeld: max β = {max(beta):.4f} → "
              f"{'satisfeito' if sass else 'NÃO satisfeito'}")
        print("      (os critérios (a) e (b) são apenas SUFICIENTES)")
        if rho is not None:
            print(f"  (c) Raio espectral da matriz de iteração de Gauss-Seidel: ρ = {rho:.4f} → "
                  f"{'ρ < 1: CONVERGE' if rho < 1 else 'ρ ≥ 1: NÃO converge'}")
        else:
            print("  (c) Raio espectral: indisponível (numpy não instalado)")
    ok = (rho < 1) if rho is not None else (dom or sass)
    return {"ok": ok, "rho": rho}


def buscar_ordenacao(cand, n):
    """
    Testa subconjuntos de n equações entre as candidatas e todas as ordens possíveis
    (a ordem define quem fica na diagonal) e devolve a de menor raio espectral < 1.
    """
    if np is None or n > 7:
        return None
    melhor_rho, melhor = math.inf, None
    for sub in itertools.combinations(range(len(cand)), n):
        if np.linalg.matrix_rank(np.array([cand[i][2] for i in sub], float)) < n:
            continue                      # equações dependentes
        for perm in itertools.permutations(sub):
            A = [cand[i][2] for i in perm]
            r = raio_espectral_gs(A)
            if r < melhor_rho:
                melhor_rho, melhor = r, list(perm)
    if melhor is not None and melhor_rho < 1 - 1e-9:
        return melhor
    return None


def equacoes_normais(A, b):
    m = len(A[0])
    AtA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(m)] for i in range(m)]
    Atb = [sum(A[k][i] * b[k] for k in range(len(A))) for i in range(m)]
    return AtA, Atb


def preparar_sistema_gs(cand, n, ordem):
    """
    Monta A·x = b na ordem dada e garante convergência:
      1) tenta a ordem dada; 2) procura outra ordem que convirja;
      3) como último recurso, usa as equações normais AᵀA·x = Aᵀb (sempre convergem).
    cand = lista de (rótulo, texto, coeficientes, termo independente).
    """
    def montar(idx):
        return ([cand[i][2][:] for i in idx], [cand[i][3] for i in idx], [cand[i][0] for i in idx])

    A, b, rot = montar(ordem)
    print("  Análise de convergência do sistema na ordem escolhida:")
    info = analisar_convergencia(A)
    if info["ok"]:
        return A, b, rot, "direto"

    print("\n  ⚠ Nesta ordenação o Gauss-Seidel NÃO converge. Procurando outra ordem das equações...")
    nova = buscar_ordenacao(cand, n)
    if nova is not None:
        A, b, rot = montar(nova)
        print("  ✓ Encontrada uma ordenação convergente. Nova análise:")
        analisar_convergencia(A)
        return A, b, rot, "reordenado"

    print("  ✗ Nenhuma ordenação das equações converge.")
    print("  → Usando as equações normais  AᵀA·x = Aᵀb  (matriz simétrica definida positiva:")
    print("    o Gauss-Seidel sempre converge, embora mais devagar).")
    An, bn = equacoes_normais([c[2] for c in (cand[i] for i in ordem)], [cand[i][3] for i in ordem])
    return An, bn, [f"AᵀA linha {i + 1}" for i in range(n)], "normais"


def detalhar_primeira_iteracao(A, b, x0, nomes):
    n = len(A)
    xn = list(x0)
    print("  Fórmula:  xᵢ⁽ᵏ⁺¹⁾ = ( bᵢ − Σ_{j<i} aᵢⱼ·xⱼ⁽ᵏ⁺¹⁾ − Σ_{j>i} aᵢⱼ·xⱼ⁽ᵏ⁾ ) / aᵢᵢ")
    print("  (usa os valores já atualizados nesta iteração — essa é a diferença para Jacobi)\n")
    for i in range(n):
        termos = []
        for j in range(n):
            if j != i and abs(A[i][j]) > 1e-15:
                origem = xn[j]                      # j<i já atualizado; j>i ainda é o antigo
                termos.append(f"({A[i][j]:g})({origem:.6f})")
        soma = " + ".join(termos) if termos else "0"
        s = sum(A[i][j] * xn[j] for j in range(n) if j != i)
        xn[i] = (b[i] - s) / A[i][i] + 0.0          # "+ 0.0" evita imprimir -0.000000
        print(f"    {nomes[i]}⁽¹⁾ = ({b[i]:g} − [{soma}]) / {A[i][i]:g} = {xn[i]:.6f}")


def executar_gauss_seidel(A, b, nomes, x0, tol, max_iter, criterio, todas=True):
    n = len(A)
    subtitulo("Primeira iteração, passo a passo (k = 0 → 1)")
    detalhar_primeira_iteracao(A, b, x0, nomes)

    larg = 11
    linhas_tab = [f"{0:>4} " + " ".join(f"{v:{larg}.6f}" for v in x0) + f" {'—':>12}"]
    x = list(x0)
    hist = []
    convergiu = False
    divergiu = False
    k = 0
    for k in range(1, max_iter + 1):
        x_ant = list(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)   # x[j<i] já novo, x[j>i] antigo
            x[i] = (b[i] - s) / A[i][i] + 0.0
        dif = max(abs(x[i] - x_ant[i]) for i in range(n))
        if criterio == "rel":
            den = max(abs(v) for v in x)
            erro = dif / den if den > 0 else dif
        else:
            erro = dif
        hist.append((k, list(x), erro))
        linhas_tab.append(f"{k:>4} " + " ".join(f"{v:{larg}.6f}" for v in x) + f" {erro:12.3e}")
        if not all(math.isfinite(v) for v in x) or max(abs(v) for v in x) > 1e50:
            divergiu = True
            break
        if erro < tol:
            convergiu = True
            break

    subtitulo("Tabela de iterações")
    print(f"{'k':>4} " + " ".join(f"{nm:>{larg}}" for nm in nomes) + f" {'erro':>12}")
    if todas or len(linhas_tab) <= 17:
        print("\n".join(linhas_tab))
    else:
        print("\n".join(linhas_tab[:11]))
        print(f"{'...':>4}   ({len(linhas_tab) - 16} iterações omitidas na exibição)")
        print("\n".join(linhas_tab[-5:]))
    if divergiu:
        print("\n  ✗ O método DIVERGIU.")
        return x, k, hist, False
    nome_crit = "erro relativo máx" if criterio == "rel" else "erro absoluto máx |xᵢ⁽ᵏ⁾ − xᵢ⁽ᵏ⁻¹⁾|"
    if convergiu:
        print(f"\n  ✓ Convergiu em {k} iterações ({nome_crit} = {hist[-1][2]:.3e} < {tol:g}).")
    else:
        print(f"\n  ✗ Não atingiu a precisão {tol:g} em {max_iter} iterações.")
    return x, k, hist, convergiu


def equacoes_wheatstone(E, R1, R2, R3, R4, R5):
    """8 equações de Kirchhoff: (rótulo, texto, coeficientes de I1..I6, termo independente)."""
    return [
        ("Nó A",           "I1 + I2 = I6",             [-1, -1, 0, 0, 0, 1], 0.0),
        ("Nó B",           "I1 = I4 + I5",             [1, 0, 0, -1, -1, 0], 0.0),
        ("Nó C",           "I2 + I5 = I3",             [0, 1, -1, 0, 1, 0], 0.0),
        ("Nó D",           "I3 + I4 = I6",             [0, 0, 1, 1, 0, -1], 0.0),
        ("Malha ABC",      "R1·I1 + R5·I5 − R2·I2 = 0", [R1, -R2, 0, 0, R5, 0], 0.0),
        ("Malha BDC",      "R4·I4 − R3·I3 − R5·I5 = 0", [0, 0, -R3, R4, -R5, 0], 0.0),
        ("Malha ABD+fonte", "R1·I1 + R4·I4 = E",       [R1, 0, 0, R4, 0, 0], float(E)),
        ("Malha ACD+fonte", "R2·I2 + R3·I3 = E",       [0, R2, R3, 0, 0, 0], float(E)),
    ]


# 6 equações independentes, na ordem que coloca cada incógnita Iᵢ na diagonal e
# faz o Gauss-Seidel convergir para os dados do enunciado (ρ ≈ 0,8828).
ORDEM_WHEATSTONE = ["Nó B", "Nó C", "Malha BDC", "Malha ABD+fonte", "Malha ABC", "Nó A"]


def grafico_convergencia(hist, nomes, titulo_graf, arquivo):
    if len(hist) < 2:
        return
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ks = [h[0] for h in hist]
    for i, nm in enumerate(nomes):
        ax[0].plot(ks, [h[1][i] for h in hist], label=nm)
    ax[0].set_xlabel("iteração k")
    ax[0].set_ylabel("valor")
    ax[0].set_title("Evolução das incógnitas")
    ax[0].legend(ncol=2, fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].semilogy(ks, [max(h[2], 1e-300) for h in hist], "o-", ms=3)
    ax[1].set_xlabel("iteração k")
    ax[1].set_ylabel("erro (escala log)")
    ax[1].set_title("Erro por iteração")
    ax[1].grid(alpha=0.3, which="both")
    fig.suptitle(titulo_graf)
    salvar_e_mostrar(plt, fig, arquivo)


def parametros_iteracao(n, tol_padrao):
    tol = ler_float("Precisão (tolerância) ε", tol_padrao)
    max_iter = ler_int("Máximo de iterações", 5000, 1, 1000000)
    crit = "abs"
    if not AUTO:
        print("  Critério de parada: [1] erro absoluto máx|x⁽ᵏ⁾−x⁽ᵏ⁻¹⁾|   [2] erro relativo")
        crit = "rel" if ler_int("Escolha", 1, 1, 2) == 2 else "abs"
    x0 = ler_lista("Aproximação inicial x⁽⁰⁾", qtd=n, padrao=[0.0] * n)
    todas = sim_nao("Mostrar todas as iterações na tabela?", True)
    return tol, max_iter, crit, x0, todas


def wheatstone():
    print("""
  Convenção da figura: nós A (topo), B (esquerda), C (direita), D (base).
  A fonte E liga D a A; I6 sobe pela fonte até A.
     I1: A→B (R1)   I2: A→C (R2)   I3: C→D (R3)   I4: B→D (R4)   I5: B→C (R5)""")
    E = ler_float("Tensão da fonte E (V)", 30)
    R = [ler_float(f"R{k} (Ω)", 20 if k == 1 else 120) for k in range(1, 6)]
    for k, r in enumerate(R, 1):
        if r <= 0:
            print(f"  ✗ R{k} deve ser positiva."); return
    nomes = [f"I{k}" for k in range(1, 7)]
    cand = equacoes_wheatstone(E, *R)
    rot_cand = [c[0] for c in cand]

    passo(1, "Equações de Kirchhoff")
    print(f"  Dados: E = {E:g} V, R1 = {R[0]:g} Ω, R2 = {R[1]:g} Ω, R3 = {R[2]:g} Ω, "
          f"R4 = {R[3]:g} Ω, R5 = {R[4]:g} Ω\n")
    print("  Lei dos nós (corrente que entra = corrente que sai):")
    for c in cand[:4]:
        print(f"     {c[0]:<16}: {c[1]}")
    print("  Lei das malhas (soma das quedas de tensão = fem):")
    for c in cand[4:]:
        print(f"     {c[0]:<16}: {c[1]}")
    print("\n  São 6 incógnitas (I1..I6) → 6 equações independentes: 3 nós (o 4º é redundante)")
    print("  e 3 malhas. As 8 equações acima são as candidatas.")

    passo(2, "Sistema linear A·I = b (ordem das equações pensada para o Gauss-Seidel)")
    print("  No Gauss-Seidel o elemento aᵢᵢ divide a atualização de xᵢ, logo precisa ser ≠ 0,")
    print("  e a ORDEM das equações decide se o método converge. Verificando:\n")
    ordem = [rot_cand.index(r) for r in ORDEM_WHEATSTONE]
    A, b, rot, modo = preparar_sistema_gs(cand, 6, ordem)
    print("\n  Sistema efetivamente usado (cada linha, uma equação):")
    if modo == "normais":
        print("   (equações normais AᵀA·I = Aᵀb)")
    for i in range(6):
        print(f"     L{i + 1} [{rot[i]}]: {eq_texto(A[i], nomes, b[i])}")
    print()
    mostrar_sistema(A, b, nomes, rotulos=[f"L{i + 1}" for i in range(6)])

    passo(3, "Método de Gauss-Seidel")
    tol, max_iter, crit, x0, todas = parametros_iteracao(6, 1e-6)
    x, its, hist, ok = executar_gauss_seidel(A, b, nomes, x0, tol, max_iter, crit, todas)
    if not ok:
        return

    passo(4, "Verificação")
    Aor = [c[2] for c in cand]
    bor = [c[3] for c in cand]
    print("  Resíduo em cada uma das 8 equações de Kirchhoff (nós em A, malhas em V),")
    print("  inclusive as 2 que não entraram no sistema resolvido:")
    for c in cand:
        r = sum(c[2][j] * x[j] for j in range(6)) - c[3]
        un = "A" if c[0].startswith("Nó") else "V"
        marca = "" if (modo == "normais" or c[0] in rot) else "   (não usada na solução)"
        print(f"     {c[0]:<16}: {r:+.2e} {un}{marca}")
    if np is not None:
        sel = [rot_cand.index(r) for r in ORDEM_WHEATSTONE]
        ref = np.linalg.solve(np.array([Aor[i] for i in sel], float),
                              np.array([bor[i] for i in sel], float))
        print(f"\n  Solução exata (numpy.linalg.solve): [" + ", ".join(f"{v:.6f}" for v in ref) + "]")
        print(f"  Erro real máx |I_GS − I_exato| = {max(abs(x[i] - ref[i]) for i in range(6)):.2e}")
    p_fonte = E * x[5]
    p_res = sum(R[k] * x[k] ** 2 for k in range(5))
    print(f"\n  Balanço de potência: P_fonte = E·I6 = {p_fonte:.6f} W   |   "
          f"Σ Rₖ·Iₖ² = {p_res:.6f} W   |   diferença = {abs(p_fonte - p_res):.2e}")

    subtitulo("RESPOSTA — correntes do circuito")
    for k in range(6):
        print(f"   {nomes[k]} = {x[k]:+.6f} A   ({x[k] * 1000:+.4f} mA)")
    print("\n   (sinal negativo = corrente no sentido contrário ao desenhado na figura)")
    if GRAFICOS:
        grafico_convergencia(hist, nomes, "Ponte de Wheatstone — Gauss-Seidel", "grafico_q02_1.png")


def gauss_seidel_generico():
    print("\n  Sistema linear qualquer A·x = b (problemas semelhantes: treliça, malhas, etc.)")
    n = ler_int("Ordem n do sistema", 3, 1, 12)
    print("  Digite cada linha da matriz A (n números separados por espaço):")
    A = [ler_lista(f"Linha {i + 1} de A", qtd=n) for i in range(n)]
    b = ler_lista("Vetor b", qtd=n)
    cand = [(f"Linha {i + 1}", "", A[i], b[i]) for i in range(n)]
    nomes = [f"x{i + 1}" for i in range(n)]
    passo(1, "Análise de convergência")
    A2, b2, rot, modo = preparar_sistema_gs(cand, n, list(range(n)))
    if modo != "direto":
        print("\n  Sistema usado:")
        mostrar_sistema(A2, b2, nomes)
    passo(2, "Método de Gauss-Seidel")
    tol, max_iter, crit, x0, todas = parametros_iteracao(n, 1e-4)
    x, its, hist, ok = executar_gauss_seidel(A2, b2, nomes, x0, tol, max_iter, crit, todas)
    if not ok:
        return
    passo(3, "Verificação")
    print(f"  Resíduo no sistema ORIGINAL: máx|A·x − b| = {residuo_max(A, x, b):.2e}")
    if np is not None:
        try:
            ref = np.linalg.solve(np.array(A, float), np.array(b, float))
            print(f"  Erro real máx |x_GS − x_exato| = {max(abs(x[i] - ref[i]) for i in range(n)):.2e}")
        except np.linalg.LinAlgError:
            print("  (A é singular: sem solução exata única para comparar)")
    subtitulo("RESPOSTA")
    for i in range(n):
        print(f"   {nomes[i]} = {x[i]:.6f}")
    if GRAFICOS:
        grafico_convergencia(hist, nomes, "Gauss-Seidel", "grafico_q02_generico.png")


def questao_02():
    titulo("SEÇÃO 2 — QUESTÃO 02.1 — MÉTODOS ITERATIVOS (GAUSS-SEIDEL): PONTE DE WHEATSTONE")
    print("  Enunciado: determinar as correntes I1..I6 da ponte de Wheatstone usando Gauss-Seidel,")
    print("  com E = 30 V, R1 = 20 Ω e R2 = R3 = R4 = R5 = 120 Ω (equações vindas de Kirchhoff).")
    print("  (O enunciado não fixa a precisão; usamos ε = 1e-6, que pode ser alterado abaixo.)")
    while True:
        if AUTO:
            modo = 1
        else:
            print("\n  [1] Ponte de Wheatstone (dados do enunciado ou novos valores de E, R1..R5)")
            print("  [2] Sistema linear qualquer A·x = b")
            modo = ler_int("Escolha", 1, 1, 2)
        if modo == 1:
            wheatstone()
        else:
            gauss_seidel_generico()
        if not sim_nao("\nFazer um novo cálculo nesta questão?", False):
            break


# =============================================================================
#  SEÇÃO 3 – QUESTÃO 03.2 – INTERPOLAÇÃO POLINOMIAL (LAGRANGE E NEWTON)
# =============================================================================
def escolher_base(xs, x, grau, modo="centrados"):
    """
    Escolhe os grau+1 pontos-base (devolve índices em ordem crescente).
      modo "proximos" : os grau+1 pontos mais próximos de x.
      modo "centrados": pontos alternando os dois lados de x (sempre o mais próximo
                        disponível do lado que tem menos pontos), para que x fique
                        centrado entre eles; sem pontos de um lado, usa só o outro.
    """
    n = grau + 1
    if modo == "proximos":
        ordem = sorted(range(len(xs)), key=lambda i: (abs(xs[i] - x), i))
        return sorted(ordem[:n])
    esq = sorted([i for i in range(len(xs)) if xs[i] <= x], key=lambda i: x - xs[i])
    dire = sorted([i for i in range(len(xs)) if xs[i] > x], key=lambda i: xs[i] - x)
    e, d = [], []
    while len(e) + len(d) < n:
        pode_e, pode_d = len(e) < len(esq), len(d) < len(dire)
        if pode_e and pode_d:
            if len(e) != len(d):
                lado = "e" if len(e) < len(d) else "d"
            else:
                lado = "e" if (x - xs[esq[len(e)]]) <= (xs[dire[len(d)]] - x) else "d"
        else:
            lado = "e" if pode_e else "d"
        if lado == "e":
            e.append(esq[len(e)])
        else:
            d.append(dire[len(d)])
    return sorted(e + d)


def produto_erro(xb, x):
    """|Π(x − xᵢ)|: fator do termo de erro da interpolação (quanto menor, melhor)."""
    p = 1.0
    for v in xb:
        p *= (x - v)
    return abs(p)


def lagrange_valor(xb, yb, x):
    tot = 0.0
    for k in range(len(xb)):
        Lk = 1.0
        for j in range(len(xb)):
            if j != k:
                Lk *= (x - xb[j]) / (xb[k] - xb[j])
        tot += yb[k] * Lk
    return tot


def diferencas_divididas(xb, yb):
    n = len(xb)
    T = [list(yb)]
    for o in range(1, n):
        ant = T[-1]
        T.append([(ant[i + 1] - ant[i]) / (xb[i + o] - xb[i]) for i in range(n - o)])
    return T


def newton_valor(xb, yb, x):
    T = diferencas_divididas(xb, yb)
    val, prod = 0.0, 1.0
    for k in range(len(xb)):
        val += T[k][0] * prod
        prod *= (x - xb[k])
    return val


def lagrange_detalhado(xb, yb, x):
    n = len(xb)
    termos = []
    soma_L = 0.0
    for k in range(n):
        fatores_txt, num, den = [], 1.0, 1.0
        for j in range(n):
            if j != k:
                fatores_txt.append(f"({x:g}−{xb[j]:g})/({xb[k]:g}−{xb[j]:g})")
                num *= (x - xb[j])
                den *= (xb[k] - xb[j])
        Lk = num / den
        soma_L += Lk
        termos.append(yb[k] * Lk)
        print(f"    L{k}(x) = " + " · ".join(fatores_txt))
        print(f"          = {num:.6f} / {den:.6f} = {Lk:.6f}")
    print("\n    Verificação: ΣLₖ(x) = " + f"{soma_L:.6f}  (deve ser 1)")
    print("    P(x) = Σ yₖ·Lₖ(x) = " + " + ".join(f"({yb[k]:g})({termos[k] / yb[k]:.6f})" if yb[k] != 0
                                                    else "0" for k in range(n)))
    print("         = " + " + ".join(f"{t:.6f}" for t in termos))
    val = sum(termos)
    print(f"         = {val:.6f}")
    return val


def newton_detalhado(xb, yb, x):
    n = len(xb)
    T = diferencas_divididas(xb, yb)
    print("    Tabela de diferenças divididas:")
    cab = f"      {'xᵢ':>8} {'f[xᵢ]':>12}" + "".join(f" {'ordem ' + str(o):>12}" for o in range(1, n))
    print(cab)
    for r in range(n):
        s = f"      {xb[r]:8g} {T[0][r]:12.6f}"
        for o in range(1, n):
            s += f" {T[o][r]:12.6f}" if r < n - o else f" {'':>12}"
        print(s)
    print("\n    Polinômio:  P(x) = f[x0] + f[x0,x1]·(x−x0) + f[x0,x1,x2]·(x−x0)(x−x1) + ...")
    print("    Coeficientes cₖ = f[x0,...,xk] = primeira linha de cada coluna: "
          + ", ".join(f"{T[k][0]:.6f}" for k in range(n)))
    val, prod = 0.0, 1.0
    print("\n    Termo a termo (soma parcial = valor do polinômio de grau k nos mesmos pontos):")
    for k in range(n):
        termo = T[k][0] * prod
        val += termo
        fat = "".join(f"(x−{xb[j]:g})" for j in range(k)) or "1"
        prod_txt = " · ".join(f"({x:g}−{xb[j]:g})" for j in range(k))
        expr = f"{T[k][0]:.6f}" + (f" · {prod_txt}" if k else "")
        print(f"      k={k}: c{k}·{fat:<22} = {expr} = {termo:+.6f}   → soma parcial = {val:.6f}")
        prod *= (x - xb[k])
    return val


def interpolar_grau(xs, ys, x, grau, modo):
    idx = escolher_base(xs, x, grau, modo)
    xb = [xs[i] for i in idx]
    yb = [ys[i] for i in idx]
    subtitulo(f"GRAU {grau}  (polinômio por {grau + 1} pontos)")
    nl = sum(1 for v in xb if v <= x)
    print(f"  Pontos-base ({'centrados em torno de' if modo == 'centrados' else 'mais próximos de'} "
          f"x = {x:g}): {nl} à esquerda e {len(xb) - nl} à direita")
    print("     xᵢ        : " + "  ".join(f"{v:8g}" for v in xb))
    print("     yᵢ        : " + "  ".join(f"{v:8g}" for v in yb))
    print("     |x − xᵢ|  : " + "  ".join(f"{abs(x - v):8.3f}" for v in xb))
    print("\n  [A] Forma de Lagrange")
    vL = lagrange_detalhado(xb, yb, x)
    print("\n  [B] Forma de Newton (diferenças divididas)")
    vN = newton_detalhado(xb, yb, x)
    print(f"\n  [C] Conferência: |Lagrange − Newton| = {abs(vL - vN):.2e}", end="")
    if np is not None:
        vp = float(np.polyval(np.polyfit(xb, yb, grau), x))
        print(f"   |   numpy.polyfit/polyval = {vp:.6f}")
    else:
        print()
    return idx, xb, yb, vL, vN


def grafico_interpolacao(xs, ys, x, resultados):
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    xx = [xs[0] + (xs[-1] - xs[0]) * t / 300 for t in range(301)]
    for grau, xb, yb, v in resultados:
        ax.plot(xx, [newton_valor(xb, yb, t) for t in xx], label=f"grau {grau}")
        ax.plot([x], [v], "s", ms=6)
    ax.plot(xs, ys, "ko", label="dados")
    ax.axvline(x, color="gray", ls=":")
    ax.set_xlabel("i (A)")
    ax.set_ylabel("V (V)")
    ax.set_title(f"Interpolação polinomial — estimativa em i = {x:g}")
    ax.legend()
    ax.grid(alpha=0.3)
    salvar_e_mostrar(plt, fig, "grafico_q03_2.png")


def resolver_interpolacao(xs, ys, x, gmin, gmax, modo="centrados"):
    passo(1, "Dados e critério de escolha dos pontos-base")
    print("     xᵢ : " + "  ".join(f"{v:8g}" for v in xs))
    print("     yᵢ : " + "  ".join(f"{v:8g}" for v in ys))
    print(f"\n  Valor a estimar: x = {x:g}")
    if modo == "centrados":
        print("  Critério adotado (pedido no enunciado): pontos-base CENTRADOS em torno de x e o mais")
        print("  próximo possível — alternando os lados de x, sempre com o ponto mais próximo disponível")
        print("  do lado que tem menos pontos.")
    else:
        print("  Critério adotado: para cada grau n, os n+1 pontos MAIS PRÓXIMOS de x.")
    if x < xs[0] or x > xs[-1]:
        print("  ⚠ x está fora do intervalo dos dados: é EXTRAPOLAÇÃO (resultado pouco confiável).")

    passo(2, "Interpolação grau a grau")
    resultados = []
    for g in range(gmin, gmax + 1):
        idx, xb, yb, vL, vN = interpolar_grau(xs, ys, x, g, modo)
        resultados.append((g, xb, yb, vL, vN, idx))

    passo(3, "Resumo")
    print(f"  {'grau':>4} | {'pontos-base (x)':<34} | {'Lagrange':>12} | {'Newton':>12} | {'|L−N|':>9}")
    linha()
    for g, xb, yb, vL, vN, idx in resultados:
        print(f"  {g:>4} | {', '.join(f'{v:g}' for v in xb):<34} | {vL:12.6f} | {vN:12.6f} | {abs(vL - vN):9.1e}")
    if len(resultados) > 1:
        print("\n  Variação entre graus consecutivos (indica a estabilidade da estimativa):")
        for a, c in zip(resultados[:-1], resultados[1:]):
            print(f"     |P{c[0]} − P{a[0]}| = {abs(c[3] - a[3]):.6f}")

    passo(4, "Sensibilidade à escolha dos pontos-base (janelas de pontos consecutivos)")
    print("  Estimativa com TODAS as janelas consecutivas de cada grau (★ = janela adotada acima).")
    print("  O erro da interpolação é proporcional a |Π(x−xᵢ)|: quanto menor, melhor a janela.\n")
    for g, xb, yb, vL, vN, idx in resultados:
        n = g + 1
        if len(xs) - n + 1 <= 1:
            print(f"  Grau {g}: só existe uma janela possível (todos os pontos).")
            continue
        print(f"  Grau {g}:")
        for i0 in range(len(xs) - n + 1):
            wx = xs[i0:i0 + n]
            wy = ys[i0:i0 + n]
            v = newton_valor(wx, wy, x)
            marca = "★" if list(range(i0, i0 + n)) == idx else " "
            print(f"    {marca} x = {{{', '.join(f'{t:g}' for t in wx)}}}   |Π(x−xᵢ)| = "
                  f"{produto_erro(wx, x):.5f}   →  P({x:g}) = {v:.6f}")

    outro = "proximos" if modo == "centrados" else "centrados"
    difs = []
    for g, xb, yb, vL, vN, idx in resultados:
        idx2 = escolher_base(xs, x, g, outro)
        if idx2 != idx:
            xb2 = [xs[i] for i in idx2]
            yb2 = [ys[i] for i in idx2]
            difs.append((g, xb2, newton_valor(xb2, yb2, x), vL))
    if difs:
        nome_outro = "pontos MAIS PRÓXIMOS" if outro == "proximos" else "pontos CENTRADOS"
        print(f"\n  ⚠ Com o outro critério ({nome_outro}) a escolha mudaria para:")
        for g, xb2, v2, v1 in difs:
            print(f"     Grau {g}: x = {{{', '.join(f'{t:g}' for t in xb2)}}}  →  {v2:.6f}  "
                  f"(diferença para o adotado: {abs(v2 - v1):.6f})")

    subtitulo(f"RESPOSTA — estimativa de V em i = {x:g} A")
    for g, xb, yb, vL, vN, idx in resultados:
        print(f"   Grau {g}: V ≈ {vL:.4f} V   (Lagrange = Newton)")
    if GRAFICOS:
        grafico_interpolacao(xs, ys, x, [(r[0], r[1], r[2], r[3]) for r in resultados])


def dados_interpolacao():
    xs_pad = [0.25, 0.75, 1.25, 1.5, 2.0]
    ys_pad = [-0.45, -0.60, 0.70, 1.88, 6.0]
    if sim_nao("Usar os dados do enunciado (corrente i × tensão V)?", True):
        return xs_pad, ys_pad, 1.15, 2, 4, escolher_modo()
    n = ler_int("Quantidade de pontos", 5, 2, 30)
    while True:
        xs = ler_lista("Valores de x (separados por espaço)", qtd=n)
        if len(set(xs)) == n:
            break
        print("   ! os valores de x devem ser todos diferentes.")
    ys = ler_lista("Valores de y", qtd=n)
    par = sorted(zip(xs, ys))
    xs = [p[0] for p in par]
    ys = [p[1] for p in par]
    x = ler_float("Valor de x a estimar")
    gmin = ler_int("Grau mínimo", min(2, n - 1), 1, n - 1)
    gmax = ler_int("Grau máximo", min(4, n - 1), gmin, n - 1)
    return xs, ys, x, gmin, gmax, escolher_modo()


def escolher_modo():
    if AUTO:
        return "centrados"
    print("  Critério de escolha dos pontos-base:")
    print("    [1] centrados em torno de x, alternando os lados (pedido do enunciado)")
    print("    [2] os n+1 pontos mais próximos de x")
    return "proximos" if ler_int("Escolha", 1, 1, 2) == 2 else "centrados"


def questao_03():
    titulo("SEÇÃO 3 — QUESTÃO 03.2 — INTERPOLAÇÃO POLINOMIAL (LAGRANGE E NEWTON)")
    print("  Enunciado: dados da queda de tensão V num resistor para várias correntes i")
    print("     i : 0.25  0.75  1.25  1.5  2.0")
    print("     V : -0.45 -0.60  0.70 1.88  6.0")
    print("  Estimar V para i = 1.15 usando as formas de Lagrange e de Newton, do 2º ao 4º grau,")
    print("  com pontos-base centrados em torno e o mais próximo possível de i = 1.15.")
    while True:
        xs, ys, x, gmin, gmax, modo = dados_interpolacao()
        resolver_interpolacao(xs, ys, x, gmin, gmax, modo)
        if not sim_nao("\nFazer um novo cálculo nesta questão?", False):
            break


# =============================================================================
#  SEÇÃO 4 – QUESTÃO 04.3 – INTEGRAÇÃO NUMÉRICA (TRAPÉZIO E SIMPSON REPETIDOS)
# =============================================================================
def trapezio_repetido(y, h):
    m = len(y) - 1
    interior = sum(y[1:m])
    return h / 2 * (y[0] + 2 * interior + y[m]), interior


def simpson13(y, h):
    """Simpson 1/3 repetido; exige número PAR de subintervalos."""
    m = len(y) - 1
    ext = y[0] + y[m]
    impares = sum(y[1:m:2])
    pares = sum(y[2:m:2])
    return h / 3 * (ext + 4 * impares + 2 * pares), ext, impares, pares


def simpson38(y, h):
    """Simpson 3/8 simples; exige exatamente 3 subintervalos (4 pontos)."""
    return 3 * h / 8 * (y[0] + 3 * y[1] + 3 * y[2] + y[3])


def pesos_simpson13(m):
    return [1] + [4 if i % 2 else 2 for i in range(1, m)] + [1]


def simpson_composto(y, h, detalhar=True):
    """
    Simpson repetido para qualquer número de subintervalos m ≥ 2:
      m par   → Simpson 1/3 repetido em todo o intervalo;
      m ímpar → Simpson 1/3 repetido nos primeiros m−3 subintervalos + Simpson 3/8 nos 3 últimos.
    """
    m = len(y) - 1
    if m % 2 == 0:
        val, ext, imp, par = simpson13(y, h)
        if detalhar:
            print(f"    m = {m} subintervalos (PAR) → Simpson 1/3 repetido em todo o intervalo")
            print(f"    pesos: {pesos_simpson13(m)}")
            print(f"    extremos y0+ym = {ext:.6f} | soma dos ímpares = {imp:.6f} | soma dos pares internos = {par:.6f}")
            print(f"    I ≈ (h/3)·[extremos + 4·ímpares + 2·pares] = ({h:g}/3)·[{ext:.6f} + 4·{imp:.6f} + 2·{par:.6f}]")
            print(f"      = ({h:g}/3)·{ext + 4 * imp + 2 * par:.6f} = {val:.6f}")
        return val
    if m == 1:
        raise ValueError("Simpson precisa de pelo menos 2 subintervalos.")
    s = m - 3
    if detalhar:
        print(f"    m = {m} subintervalos (ÍMPAR): Simpson 1/3 repetido exige m par.")
        print(f"    Solução: Simpson 1/3 em x0..x{s} ({s} subintervalos) + Simpson 3/8 em x{s}..x{m} (3 subintervalos).")
    v13 = 0.0
    if s > 0:
        v13, ext, imp, par = simpson13(y[:s + 1], h)
        if detalhar:
            print(f"\n    (i) Simpson 1/3 repetido em [x0, x{s}]  — pesos {pesos_simpson13(s)}")
            print(f"        extremos = {ext:.6f} | ímpares = {imp:.6f} | pares = {par:.6f}")
            print(f"        I₁ = ({h:g}/3)·[{ext:.6f} + 4·{imp:.6f} + 2·{par:.6f}] = {v13:.6f}")
    v38 = simpson38(y[s:], h)
    if detalhar:
        q = y[s:]
        print(f"\n    ({'ii' if s > 0 else 'i'}) Simpson 3/8 em [x{s}, x{m}]  — pesos [1, 3, 3, 1]")
        print(f"        I₂ = (3·{h:g}/8)·[{q[0]:.6f} + 3·{q[1]:.6f} + 3·{q[2]:.6f} + {q[3]:.6f}] = {v38:.6f}")
        print(f"\n    I ≈ I₁ + I₂ = {v13:.6f} + {v38:.6f} = {v13 + v38:.6f}")
    return v13 + v38


def variantes_simpson(y, h):
    """Outras formas de tratar m ímpar (só para comparação)."""
    m = len(y) - 1
    out = []
    if m % 2 == 1 and m >= 3:
        s = m - 3
        out.append(("Simpson 1/3 nos 1ºs m−3 + 3/8 nos 3 últimos (adotada)",
                    (simpson13(y[:s + 1], h)[0] if s > 0 else 0.0) + simpson38(y[s:], h)))
        out.append(("Simpson 3/8 nos 3 primeiros + 1/3 nos demais",
                    simpson38(y[:4], h) + (simpson13(y[3:], h)[0] if m - 3 > 0 else 0.0)))
        out.append(("Simpson 1/3 nos m−1 primeiros + Trapézio no último",
                    simpson13(y[:m], h)[0] + h / 2 * (y[m - 1] + y[m])))
    return out


def grafico_integracao(x, y, area_t, area_s):
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(x, y, alpha=0.25, label=f"área ≈ {area_s:.4f}")
    ax.plot(x, y, "o-", label="dados")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Integração numérica — Trapézio = {area_t:.4f} | Simpson = {area_s:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    salvar_e_mostrar(plt, fig, "grafico_q04_3.png")


def resolver_integracao(y, h, x0, unidade_area):
    m = len(y) - 1
    x = [x0 + i * h for i in range(m + 1)]

    passo(1, "Dados (pontos igualmente espaçados)")
    print(f"  Passo h = {h:g}  |  pontos: {m + 1}  |  subintervalos: m = {m}")
    print(f"  {'i':>3} | {'xᵢ':>8} | {'yᵢ':>10}")
    linha()
    for i in range(m + 1):
        print(f"  {i:>3} | {x[i]:8.3f} | {y[i]:10.4f}")

    passo(2, "Regra do Trapézio repetida")
    val_t, interior = trapezio_repetido(y, h)
    print("    I ≈ (h/2)·[y0 + 2·(y1 + ... + y_{m−1}) + y_m]")
    print(f"    soma dos pontos internos = {interior:.6f}")
    print(f"    I ≈ ({h:g}/2)·[{y[0]:.4f} + 2·{interior:.6f} + {y[m]:.4f}] = "
          f"{h / 2:g}·{y[0] + 2 * interior + y[m]:.6f} = {val_t:.6f}")

    passo(3, "Regra de Simpson repetida")
    val_s = None
    if m < 2:
        print("    Simpson não se aplica com apenas 1 subintervalo.")
    else:
        val_s = simpson_composto(y, h)

    passo(4, "Comparação e conferências")
    print(f"  Trapézio repetido : {val_t:.6f} {unidade_area}")
    if val_s is not None:
        print(f"  Simpson repetido  : {val_s:.6f} {unidade_area}")
        print(f"  Diferença |T − S| : {abs(val_t - val_s):.6f}  ({abs(val_t - val_s) / abs(val_s) * 100:.2f}% do valor de Simpson)")
        extras = variantes_simpson(y, h)
        if extras:
            print("\n  Como o nº de subintervalos é ímpar, outras formas de combinar as regras dariam:")
            for nome, v in extras:
                print(f"     {nome:<58} {v:.6f}")
    if np is not None:
        f = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        print(f"\n  Conferência do Trapézio com numpy.trapezoid: {float(f(y, dx=h)):.6f}")

    subtitulo("RESPOSTA")
    print(f"   Área pelo Trapézio repetido: {val_t:.4f} {unidade_area}")
    if val_s is not None:
        print(f"   Área por Simpson repetido  : {val_s:.4f} {unidade_area}")
    if GRAFICOS:
        grafico_integracao(x, y, val_t, val_s if val_s is not None else val_t)


def dados_integracao():
    y_pad = [3.00, 2.92, 2.75, 2.52, 2.30, 1.84, 0.92, 0.0]
    if sim_nao("Usar os dados do enunciado (seção do navio, A..A7)?", True):
        return y_pad, 0.4, 0.0, "m²"
    h = ler_float("Espaçamento constante h entre os pontos")
    if h <= 0:
        print("   ! h deve ser positivo; usando h = 1.")
        h = 1.0
    y = ler_lista("Valores yᵢ (largura/profundidade/função em cada ponto)")
    x0 = ler_float("Valor de x no primeiro ponto", 0.0)
    return y, h, x0, "(unid. de x · unid. de y)"


def questao_04():
    titulo("SEÇÃO 4 — QUESTÃO 04.3 — INTEGRAÇÃO NUMÉRICA (TRAPÉZIO E SIMPSON REPETIDOS)")
    print("  Enunciado: área da seção do trecho mais largo de um navio. Larguras medidas a cada")
    print("  0,4 m de profundidade (A, A1, ..., A7):")
    print("     3.00  2.92  2.75  2.52  2.30  1.84  0.92  (e 0 em A7, onde a curva toca o eixo)")
    print("  Calcular a área com as Regras do Trapézio e de Simpson repetidas.")
    while True:
        y, h, x0, un = dados_integracao()
        if len(y) < 2:
            print("   ! são necessários pelo menos 2 pontos.")
        else:
            resolver_integracao(y, h, x0, un)
            print("\n  Observação: a área calculada é a da região da figura (entre a linha vertical A–A7")
            print("  e o casco). Se a linha vertical for o eixo de simetria do navio, a seção completa")
            print("  tem o dobro desse valor.")
        if not sim_nao("\nFazer um novo cálculo nesta questão?", False):
            break


# =============================================================================
#  MENU PRINCIPAL
# =============================================================================
def menu():
    global GRAFICOS
    while True:
        print()
        linha("=")
        print(" CÁLCULO NUMÉRICO — ATIVIDADE COMPLEMENTAR (LES710C)")
        linha("=")
        print("  [1] Questão 01.1 — Sistemas lineares: métodos diretos (produção de componentes)")
        print("  [2] Questão 02.1 — Sistemas lineares: Gauss-Seidel (Ponte de Wheatstone)")
        print("  [3] Questão 03.2 — Interpolação de Lagrange e de Newton")
        print("  [4] Questão 04.3 — Integração numérica: Trapézio e Simpson repetidos")
        print(f"  [5] Gráficos: {'LIGADOS' if GRAFICOS else 'desligados'} (digite 5 para alternar)")
        print("  [0] Sair")
        op = _entrada("\n  Escolha uma opção: ").strip()
        if op == "1":
            questao_01()
        elif op == "2":
            questao_02()
        elif op == "3":
            questao_03()
        elif op == "4":
            questao_04()
        elif op == "5":
            GRAFICOS = not GRAFICOS
        elif op == "0":
            print("\n  Encerrando. Bom trabalho!\n")
            return
        else:
            print("  Opção inválida.")


def main():
    if AUTO:
        for q in (questao_01, questao_02, questao_03, questao_04):
            q()
    else:
        try:
            menu()
        except KeyboardInterrupt:
            print("\n\n  Programa interrompido pelo usuário.\n")


if __name__ == "__main__":
    main()