import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time
from random import uniform, choice, random, randint
from copy import deepcopy

# --- Implementação do Black Widow Optimization (BWO) de nathanrooy/bwo ---
# Adaptado para este script.

def _generate_new_position(x0: list = None, dof: int = None, bounds: list = None) -> list:
    if x0 and bounds:
        return [min(max(uniform(-1, 1) + x0[i], bounds[i][0]), bounds[i][1]) for i in range(len(x0))]
    if bounds:
        return [uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    if x0:
        return [x_i + uniform(-1, 1) for x_i in x0]
    if dof:
        return [uniform(-1, 1) for _ in range(0, dof)]

def black_widow_optimization(func, dof, bounds, npop, maxiter, pp, cr, pm, disp=False, output_lines=None):
    """
    Executa o algoritmo Black Widow Optimization para encontrar o mínimo de uma função.

    Metaparâmetros (baseado em nathanrooy/bwo):
    - func: A função objetivo a ser minimizada.
    - dof (int): Graus de liberdade (neste caso, o número de atributos).
    - bounds (list): Uma lista de tuplas definindo os limites para cada dimensão. Ex: [(0, 1), (0, 1), ...].
    - npop (int): Tamanho da população.
    - maxiter (int): Número máximo de iterações.
    - pp (float): Percentual de procriação (procreating percentage). Controla o tamanho da população que se reproduz.
    - cr (float): Taxa de canibalismo (cannibalism rate). Na verdade, representa a taxa de sobrevivência dos filhotes.
                   Um valor de 1 significa que todos os filhotes sobrevivem.
    - pm (float): Taxa de mutação (mutation rate).
    - disp (bool): Se True, mostra o progresso a cada iteração.
    - output_lines (list): Lista para armazenar as linhas de saída.
    """
    nr = int(npop * pp)
    nm = int(npop * pm)
    spacer = len(str(npop))

    pop = [_generate_new_position(dof=dof, bounds=bounds) for _ in range(0, npop)]
    
    gbest_val = float('inf')
    gbest_pos = None

    for epoch in range(0, maxiter):
        pop = sorted(pop, key=lambda x: func(x), reverse=False)
        
        current_best_val = func(pop[0])
        if current_best_val < gbest_val:
            gbest_val = current_best_val
            gbest_pos = pop[0]

        log_line = f'> Iteração: {epoch+1:>{spacer}} | Melhor Custo (1-acc): {gbest_val:0.6f}'
        if disp:
            print(log_line)
        if output_lines is not None:
            output_lines.append(f'`{log_line}`')


        pop1 = deepcopy(pop[:nr])
        pop2 = []
        pop3 = []

        for i in range(0, nr):
            try:
                i1, i2 = randint(0, len(pop1)-1), randint(0, len(pop1)-1)
                p1, p2 = pop1[i1], pop1[i2]
            except (ValueError, IndexError):
                continue # Pula se a população de pais estiver vazia

            children = []
            for _ in range(0, int(dof/2)):
                alpha = random()
                c1 = [(alpha * v1) + ((1 - alpha)*v2) for v1, v2 in zip(p1, p2)]
                c2 = [(alpha * v2) + ((1 - alpha)*v1) for v1, v2 in zip(p1, p2)]
                children.append(c1)
                children.append(c2)

            if func(p1) > func(p2):
                if i1 < len(pop1): pop1.pop(i1)
            else:
                if i2 < len(pop1): pop1.pop(i2)

            children = sorted(children, key=lambda x: func(x), reverse=False)
            children = children[:max(int(len(children) * cr), 1)]
            pop2.extend(children)

        for _ in range(0, nm):
            if not pop2: continue
            m = choice(pop2)
            cp1, cp2 = randint(0, dof-1), randint(0, dof-1)
            m[cp1], m[cp2] = m[cp2], m[cp1]
            pop3.append(m)

        pop2.extend(pop3)
        if pop2:
            pop = deepcopy(pop2)

    # Garante que uma melhor posição seja retornada mesmo que a otimização falhe
    if gbest_pos is None and pop:
        gbest_pos = pop[0]

    return gbest_val, gbest_pos

# --- Função Objetivo e Carregamento de Dados ---

def create_objective_function(X, y, threshold=0.5):
    """
    Cria uma função objetivo para o BWO. A função avalia um subconjunto de atributos
    calculando a acurácia de um classificador k-NN e retorna (1 - acurácia).
    """
    def objective_function(solution):
        binary_solution = [1 if x > threshold else 0 for x in solution]
        selected_features_indices = np.where(np.array(binary_solution) == 1)[0]

        if len(selected_features_indices) == 0:
            return 1.0

        X_subset = X[:, selected_features_indices]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        return 1.0 - accuracy
        
    return objective_function

def load_arff_file(file_path):
    """Carrega um arquivo .arff, trata dados categóricos e o prepara para o scikit-learn."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data_dic = arff.load(f)

    attributes = [attr[0] for attr in data_dic['attributes']]
    df = pd.DataFrame(data_dic['data'], columns=attributes)

    X_raw = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_encoded = pd.get_dummies(X_raw)

    return X_encoded.values, y, X_encoded.columns.tolist()

# --- Execução Principal ---

def main():
    arff_files = ['Base_Mat..arff', 'Base_Port..arff']
    
    NPOP = 20
    MAXITER = 30
    PP = 0.6
    CR = 0.44
    PM = 0.4

    output_lines = ["# Análise de Otimização Black Widow"]

    for file_path in arff_files:
        print(f"\n{'='*20} Analisando o arquivo: {file_path} {'='*20}")
        output_lines.append(f"\n## Analisando o arquivo: `{file_path}`\n")
        
        try:
            X, y, attributes = load_arff_file(file_path)
            num_features = X.shape[1]
            
            log_line = f"Arquivo carregado e processado: {X.shape[0]} instâncias, {num_features} atributos (após one-hot encoding)."
            print(log_line)
            output_lines.append(log_line)

            bounds = [(0, 1)] * num_features
            objective_func = create_objective_function(X, y, threshold=0.5)

            print("\nIniciando otimização BWO com os seguintes metaparâmetros:")
            output_lines.append("\n### Metaparâmetros da Otimização BWO")
            output_lines.append(f"- **Tamanho da População (npop):** `{NPOP}`")
            output_lines.append(f"- **Gerações (maxiter):** `{MAXITER}`")
            output_lines.append(f"- **% de Procriação (pp):** `{PP}`")
            output_lines.append(f"- **Taxa de Sobrevivência de Filhotes (cr):** `{CR}`")
            output_lines.append(f"- **Taxa de Mutação (pm):** `{PM}`\n")
            print(f"- Tamanho da População (npop): {NPOP}")
            print(f"- Gerações (maxiter): {MAXITER}")
            print(f"- % de Procriação (pp): {PP}")
            print(f"- Taxa de Sobrevivência de Filhotes (cr): {CR}")
            print(f"- Taxa de Mutação (pm): {PM}\n")

            output_lines.append("### Log de Iterações\n```")
            start_time = time.time()
            best_cost, best_solution = black_widow_optimization(
                func=objective_func,
                dof=num_features,
                bounds=bounds,
                npop=NPOP,
                maxiter=MAXITER,
                pp=PP,
                cr=CR,
                pm=PM,
                disp=True,
                output_lines=output_lines
            )
            end_time = time.time()
            output_lines.append("```")

            if best_solution is None:
                log_line = "\n--- A otimização não encontrou uma solução válida. ---"
                print(log_line)
                output_lines.append(f"\n**{log_line.strip()}**")
                continue

            final_accuracy = 1.0 - best_cost
            binary_solution = [1 if x > 0.5 else 0 for x in best_solution]
            selected_features_indices = np.where(np.array(binary_solution) == 1)[0]
            selected_features_names = [attributes[i] for i in selected_features_indices]

            print("\n--- Resultados da Otimização BWO ---")
            output_lines.append("\n### Resultados da Otimização BWO")

            log_line = f"Tempo de execução: {end_time - start_time:.2f} segundos"
            print(log_line)
            output_lines.append(f"- **Tempo de execução:** `{end_time - start_time:.2f} segundos`")

            log_line = f"Melhor acurácia encontrada: {final_accuracy:.4f}"
            print(log_line)
            output_lines.append(f"- **Melhor acurácia encontrada:** `{final_accuracy:.4f}`")

            log_line = f"Número de atributos selecionados: {len(selected_features_names)}"
            print(log_line)
            output_lines.append(f"- **Número de atributos selecionados:** `{len(selected_features_names)}`")

            print("Atributos selecionados:")
            output_lines.append("- **Atributos selecionados:**")
            if selected_features_names:
                for name in selected_features_names:
                    print(f"- {name}")
                    output_lines.append(f"  - `{name}`")
            else:
                log_line = "Nenhum atributo foi selecionado pela otimização."
                print(log_line)
                output_lines.append(f"  - {log_line}")

        except FileNotFoundError:
            log_line = f"ERRO: Arquivo '{file_path}' não encontrado. Verifique o nome e o caminho."
            print(log_line)
            output_lines.append(f"**ERRO:** Arquivo `{file_path}` não encontrado. Verifique o nome e o caminho.")
        except Exception as e:
            log_line = f"Ocorreu um erro ao processar o arquivo {file_path}: {e}"
            print(log_line)
            output_lines.append(f"**ERRO:** Ocorreu um erro ao processar o arquivo `{file_path}`: {e}")

    with open("analise_bwo_output.md", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print("\nAnálise concluída. Resultados salvos em 'analise_bwo_output.md'")


if __name__ == "__main__":
    main()
