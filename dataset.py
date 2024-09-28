import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, img_dir, ann_dir, max_boxes=10, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.max_boxes = max_boxes
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')

        # Obtener el nombre base de la imagen (sin la extension por previos errores de numeros y simbolos)
        base_name = os.path.splitext(self.img_names[idx])[0]
        
        # Buscar el archivo de anotacion que comienza con el nombre base de la imagen
        anno_files = [f for f in os.listdir(self.ann_dir) if f.startswith(base_name)]
        
        # Verificar si se encontro algun archivo de anotación
        if len(anno_files) == 0:
            raise FileNotFoundError(f'No annotation file found for image: {img_name}')
        
        # Tomar el primer archivo de anotación encontrado
        anno_name = os.path.join(self.ann_dir, anno_files[0])
        print(f'Loading annotation: {anno_name}')  # Opcion de depuracion 

        # Obtener las cajas y etiquetas de la anotacion
        boxes, labels = self.parse_annotation(anno_name)

        # Rellenar o truncar las cajas y etiquetas
        if len(boxes) < self.max_boxes:
            pad_amount = self.max_boxes - len(boxes)
            boxes = torch.cat([boxes, torch.zeros((pad_amount, boxes.size(1)))], dim=0)
            labels = torch.cat([labels, torch.zeros(pad_amount, dtype=torch.long)], dim=0)
        else:
            boxes = boxes[:self.max_boxes]
            labels = labels[:self.max_boxes]

        if self.transform:
            image = self.transform(image)

        return image, (boxes, labels)

    def parse_annotation(self, annotation_file):
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        boxes = []
        labels = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            boxes.append([x_center, y_center, width, height])
            labels.append(class_id)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
