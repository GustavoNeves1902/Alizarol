import os
import argparse
import torch
import torch.utils.data
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm

# --- 1. FUNÇÕES DE CARREGAMENTO E CONFIGURAÇÃO (Adaptadas do train_CNN.py) ---

# Adaptação do transform_images para o conjunto de testes
def transform_images_test():
    # Usamos a mesma transformação de redimensionamento forçado 
    # (para garantir 224x224)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])
    return data_transform

# Adaptação do setting_model (simplificado para carregar a arquitetura)
def setting_model(cnn_model_name):
    # As arquiteturas são carregadas sem pesos pré-treinados,
    # pois o modelo treinado será carregado em seguida.
    if cnn_model_name == 'resnet50':
        cnn_model = models.resnet50(weights=None)
        num_features = cnn_model.fc.in_features
        cnn_model.fc = torch.nn.Linear(num_features, 2)
    elif cnn_model_name == 'vgg16':
        cnn_model = models.vgg16(weights=None)
        num_features = cnn_model.classifier[6].in_features
        cnn_model.classifier[6] = torch.nn.Linear(num_features, 2)
    elif cnn_model_name == 'googlenet':
        cnn_model = models.googlenet(weights=None)
        num_features = cnn_model.fc.in_features
        cnn_model.fc = torch.nn.Linear(num_features, 2)
    else:
        print('Invalid model')
        sys.exit(1)
        
    return cnn_model

# --- 2. FUNÇÃO PRINCIPAL DE AVALIAÇÃO ---

def evaluate_model(exp_name, model_name, batch_size=32):
    
    # Define o dispositivo (CPU ou CUDA, se o PyTorch do servidor for o correto)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. CARREGAR DADOS DE TESTE
    data_transform = transform_images_test()
    test_folder = os.path.join('tests', exp_name)
    test_datasets = datasets.ImageFolder(test_folder, transform=data_transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, shuffle=False, num_workers=0 # num_workers=0 por segurança
    )

    # 2. CARREGAR MODELO TREINADO
    # O script de treino salva o nome do modelo sem os hiperparâmetros.
    weights_filename = f'{model_name}_{exp_name}.pth'
    model_path = os.path.abspath(os.path.join('models', exp_name, weights_filename))

    if not os.path.exists(model_path):
        print(f"Erro: Arquivo de pesos não encontrado em {model_path}")
        print("Certifique-se de que o treinamento com train_CNN.py foi concluído com sucesso.")
        return

    # Inicializa a arquitetura (sem DataParallel, pois DataParallel é opcional na inferência)
    model = setting_model(model_name)
    
    # Carrega os pesos salvos
    try:
        # Nota: O .pth salva o state_dict do DataParallel, então é necessário 
        # ajustar os nomes das chaves antes de carregar no modelo sem DataParallel.
        state_dict = torch.load(model_path, map_location=device)
        
        # Remove o prefixo 'module.' das chaves se o modelo foi salvo usando DataParallel
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(new_state_dict)
        model.to(device)
    except Exception as e:
        print(f"Erro ao carregar os pesos do modelo: {e}")
        return
        

    # 3. EXECUTAR AVALIAÇÃO
    model.eval()
    all_predictions = []
    all_labels = []

    print(f"\nIniciando avaliação do modelo {model_name}...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc='Avaliando'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            # Para GoogleNet sem pré-treinamento, pode haver .logits, mas no eval, 
            # geralmente é desnecessário se o modelo for bem carregado.

            # Obtém a previsão (classe com maior score)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 4. CALCULAR MÉTRICAS
    
    # Converte listas para NumPy arrays para uso com scikit-learn
    y_true = all_labels
    y_pred = all_predictions

    results = {
        'Acurácia': accuracy_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred),
        'Recall (Sensibilidade)': recall_score(y_true, y_pred)
    }

    # 5. EXIBIR RESULTADOS
    print("\n--- Resultados de Classificação Binária ---")
    print(f"Modelo: {model_name}")
    print(f"Conjunto de Teste: {exp_name}")
    print("------------------------------------------")
    
    for metric, value in results.items():
        print(f"{metric:<25}: {value:.4f}")

    print("------------------------------------------")
    
    # 6. SALVAR RESULTADOS (Opcional, mas recomendado)
    results_df = pd.DataFrame([results])
    results_path = os.path.abspath(f'results/{exp_name}')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_file = os.path.join(results_path, f'Evaluation_{model_name}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResultados salvos em: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Avalia o modelo CNN treinado.')
    parser.add_argument('-e', '--experiment', type=str, required=True, help='Nome do experimento (ex: Alizarol_Leite)')
    parser.add_argument('-m', '--model', default='resnet50', type=str, help='Modelo CNN (ex: resnet50)')
    args = parser.parse_args()

    evaluate_model(args.experiment, args.model)
