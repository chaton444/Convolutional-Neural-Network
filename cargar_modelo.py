# cargar_modelo.py
import torch
from modelo import ObjectDetector

# Parámetros
num_classes = 1  # Ajusta según el número de clases
save_path = 'object_detector.pth'  # Ruta donde está guardado el modelo

# Inicializar el modelo
model = ObjectDetector(num_classes=num_classes)

# Cargar los pesos guardados
model.load_state_dict(torch.load(save_path))

# Poner el modelo en modo evaluación (importante para la inferencia)
model.eval()

# Ahora puedes usar el modelo para inferencia sin necesidad de volver a entrenarlo
