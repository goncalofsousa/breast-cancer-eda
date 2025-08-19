import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Stores all results used in functions
results = {}

def load_data():
    try:
        path_file = input('Enter the file path: ')
        data = pd.read_csv(path_file)
        results['load_data'] = data
        return data
    except FileNotFoundError:
        print('File not found')
        return None

def data_info(data):
    if data is not None:
        import sys
        from io import StringIO

        print("Information about columns and data types:")
        buffer = StringIO()
        sys.stdout = buffer  # Redirect output to buffer
        data.info()
        sys.stdout = sys.__stdout__  # Restore standard output

        col_info = buffer.getvalue()
        results['data_info'] = col_info
        print(col_info)

        null_values = data.isnull().sum()
        results['null_values'] = null_values
        print("Null values per column:")
        print(null_values)
        
        if null_values.any():
            print("WARNING: Null values found.")
            print("Options to handle null values:")
            print("1. Fill with a specific value")
            print("2. Remove rows/columns with null values")
            option = input("Choose an option to handle null values (1/2): ")
            if option == '1':
                fill_value = input("Enter the value to fill null values: ")
                data = data.fillna(fill_value)
                print(f"Null values filled with '{fill_value}'.")
            elif option == '2':
                data = data.dropna()
                print("Rows/columns with null values have been removed.")
            else:
                print("Invalid option.")
    else:
        print('No data loaded')

def show_classes_and_genes(data):
    if data is not None:
        cancer_classes = data['type'].unique()
        print("Cancer classes in the dataset:")
        print(cancer_classes)
        
        selected_class = input("Choose a cancer class to view the corresponding genes: ")
        if selected_class in cancer_classes:
            class_data = data[data['type'] == selected_class]
            genes = class_data.drop(columns=['samples', 'type'])
            gene_names = genes.columns.tolist()
            results[f'genes_by_class_{selected_class}'] = gene_names
            print(f"Gene names for class '{selected_class}':")
            print(gene_names)
        else:
            print(f"The class '{selected_class}' is not in the dataset.")
    else:
        print("No data loaded.")

def plot_gene_distribution_by_class(data):
    if data is not None:
        gene_options = data.drop(columns=['samples', 'type']).columns.tolist()
        class_options = data['type'].unique().tolist()
        gene = input("Enter the gene name to visualize the distribution: ")
        cancer_class = input("Enter the cancer class to visualize the distribution: ")
        
        if gene in gene_options and cancer_class in class_options:
            plt.figure(figsize=(8, 6))
            plt.hist(data[data['type'] == cancer_class][gene], alpha=0.5)
            plt.xlabel('Gene Expression')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of gene "{gene}" in class "{cancer_class}"')
            plt.show()
            results['plot_gene_distribution_by_class'] = plt.gcf()
        else:
            print("Gene or class not found.")
    else:
        print("No data loaded.")

def compare_gene_expression_across_classes(data):
    if data is not None:
        gene_options = data.drop(columns=['samples', 'type']).columns.tolist()
        class_options = data['type'].unique().tolist()
        
        genes = input("Enter gene names to compare (space separated): ").split()
        classes = input("Enter class names to compare (space separated): ").split()
        
        if all(g in gene_options for g in genes) and all(c in class_options for c in classes):
            plt.figure(figsize=(10, 8))
            for c in classes:
                class_data = data[data['type'] == c]
                for g in genes:
                    plt.scatter(class_data['samples'], class_data[g], label=f'{g} - {c}')
            plt.xlabel('Samples')
            plt.ylabel('Gene Expression')
            plt.title(f'Comparison of genes {", ".join(genes)} across classes')
            plt.legend()
            plt.show()
            results['compare_gene_expression_across_classes'] = plt.gcf()
        else:
            print("Invalid genes or classes.")
    else:
        print("No data loaded.")

def correlation_matrix_heatmap(data):
    if data is not None:
        genes = data.drop(columns=['samples', 'type']).columns.tolist()
        selected_genes = input("Enter genes for analysis (space separated): ").split()

        gene_data = data.drop(columns=['samples', 'type'])
        if selected_genes:
            gene_data = gene_data[selected_genes]

        corr_matrix = gene_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
        plt.title('Correlation Matrix of Genes')
        plt.show()
        results['correlation_matrix_heatmap'] = plt.gcf()
    else:
        print("No data loaded.")

def cluster_visualization_pca(data):
    if data is not None:
        print("Available cancer classes:")
        class_options = data['type'].unique().tolist()
        print(class_options)
        
        selected_class = input("Choose a cancer class for analysis: ")
        n_clusters = int(input("Enter number of clusters: "))

        if selected_class in class_options:
            filtered_data = data[data['type'] == selected_class].drop(columns=['samples', 'type'])
            
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(filtered_data)
            
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(filtered_data)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'Cluster Visualization for "{selected_class}" with {n_clusters} clusters (PCA)')
            plt.show()
            results['cluster_visualization_pca'] = plt.gcf()
        else:
            print("Class not found.")
    else:
        print("No data loaded.")

def identify_important_genes(data, results):
    if data is not None:
        print("Available cancer classes:")
        class_options = data['type'].unique().tolist()
        print(class_options)
        
        selected_class = input("Choose a cancer class for analysis: ")
        if selected_class in class_options:
            class_data = data[data['type'] == selected_class]
            X = class_data.drop(columns=['samples', 'type'])
            y = class_data['type']
            
            n_genes = int(input("Enter number of top genes to select: "))

            k_best = SelectKBest(score_func=f_classif, k=n_genes)
            X_best = k_best.fit_transform(X, y)
            
            selected_indices = k_best.get_support(indices=True)
            selected_genes = X.columns[selected_indices]
            
            print(f"Top {n_genes} genes relevant for {selected_class}:")
            print(selected_genes)
            
            results[f'important_genes_{selected_class}'] = selected_genes.tolist()
        else:
            print("Class not found.")
    else:
        print("No data loaded.")

def save_results():
    for func_name, result in results.items():
        if isinstance(result, pd.DataFrame):
            result.to_csv(f"{func_name}.csv", index=False)
        elif isinstance(result, str):
            with open(f"{func_name}.txt", "w") as file:
                file.write(result)
        elif isinstance(result, plt.Figure):
            plt.figure(result.number)
            plt.savefig(f"{func_name}.png", dpi=300)
        elif func_name == 'data_info':
            with open(f"{func_name}.txt", "w") as file:
                file.write(str(result))
        elif func_name.startswith('genes_by_class_'):
            class_name = func_name.split('_')[-1]
            with open(f"{func_name}.txt", "w") as file:
                genes = ', '.join(result)
                file.write(f"Genes for class '{class_name}': {genes}")
        elif func_name.startswith('important_genes_'):
            class_name = func_name.split('_')[-1]
            with open(f"{func_name}.txt", "w") as file:
                genes = ', '.join(result)
                file.write(f"Important genes for '{class_name}': {genes}")
        else:
            print(f"Could not save result for '{func_name}'")
