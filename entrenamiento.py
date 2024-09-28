# entrenamiento de la red neuronal convolucional
# Fernando Leon Medina /chaton444
import torch
import torch.optim as optim
from modelo import DetectordeObjetos
from dataset import CustomDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Parametros del entrenamiento
num_classes = 1  # Ajusta segun el numero de clases
num_epochs = 60 # cuantas epocas quiero que se ejecuten
batch_size = 16 # el tamaño del batch
learning_rate = 0.001 # aqui ajusto la tasa de aprendizaje

# Preparar datos redimencionando mis imagenes al formato de mi red neuronal convolucional
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
#aqui se pone la ruta de el train  y los vals para poder entrenar  con bounding boxes(desde la funcion hechar en otro python)
train_dataset = CustomDataset('C:/Users/chato/OneDrive/Escritorio/ProyectoTitulacion/data/images/train', 'C:/Users/chato/OneDrive/Escritorio/ProyectoTitulacion/data/images/labels', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, max_boxes=10))
#aqui ayuda a que se delimite los cuadros delimitadores y realiza un solo tensor a base de las imagenes 
def collate_fn(batch, max_boxes=10):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)


    padded_boxes = []
    padded_labels = []
    for boxes, labels in targets:
        if len(boxes) < max_boxes:
            pad_amount = max_boxes - len(boxes)
            boxes = torch.cat([boxes, torch.zeros((pad_amount, boxes.size(1)))], dim=0)
            labels = torch.cat([labels, torch.zeros(pad_amount, dtype=torch.long)], dim=0)
        else:
            boxes = boxes[:max_boxes]
            labels = labels[:max_boxes]
        padded_boxes.append(boxes)
        padded_labels.append(labels)

    padded_boxes = torch.stack(padded_boxes, 0)
    padded_labels = torch.stack(padded_labels, 0)

    return images, (padded_boxes, padded_labels)

# Inicializar modelo, criterio y optimizador
model = DetectordeObjetos(num_classes=num_classes)
criterion = torch.nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Adam para actualizar los pesos de la red y optimizar

# Entrenamiento
def train_model(model, data_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, (boxes, labels) in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, boxes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(data_loader)}')

        # Guardar el modelo y el optimizador cada epoca
        torch.save(model.state_dict(), 'model.pth')
        torch.save(optimizer.state_dict(), 'optimizer.pth')

train_model(model, train_loader, optimizer, criterion, num_epochs)
