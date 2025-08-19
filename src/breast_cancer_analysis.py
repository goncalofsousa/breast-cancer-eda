
from utils import load_data, data_info, show_classes_and_genes, plot_gene_distribution_by_class, compare_gene_expression_across_classes, correlation_matrix_heatmap, cluster_visualization_pca, identify_important_genes, save_results, results

def menu():
    data = None
    while True:
        print("----- Main Menu -----")
        print("1. Load data")
        print("2. Data information")
        print("3. Show classes and genes")
        print("4. Plot gene distribution by class")
        print("5. Compare gene expression across classes")
        print("6. Correlation matrix and Heatmap")
        print("7. Cluster visualization with PCA")
        print("8. Identify important genes")
        print("9. Save results")
        print("0. Exit")

        choice = input("Choose an option (0-9): ")
        
        if choice == '1':
            data = load_data()
        elif choice == '2':
            data_info(data)
        elif choice == '3':
            show_classes_and_genes(data)
        elif choice == '4':
            plot_gene_distribution_by_class(data)
        elif choice == '5':
            compare_gene_expression_across_classes(data)
        elif choice == '6':
            correlation_matrix_heatmap(data)
        elif choice == '7':
            cluster_visualization_pca(data)
        elif choice == '8':
            identify_important_genes(data, results)
        elif choice == '9':
            save_results()
        elif choice == '0':
            print("Exiting program...")
            break
        else:
            print("Invalid option. Choose between 0 and 9.")

menu()