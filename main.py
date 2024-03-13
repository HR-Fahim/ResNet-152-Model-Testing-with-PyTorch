import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained ResNet-50 model
model = models.resnet152(pretrained=True)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess sample image
image_path = 'self-tested\\images\\Cat.jpg'
image = Image.open(image_path)
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# Perform inference
with torch.no_grad():
    output = model(input_batch)

# Process the output
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the class labels
with open('self-tested\\imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# Print the top 5 predicted classes with probabilities in percentage
top5_prob, top5_indices = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(labels[top5_indices[i]], "{:.2f}%".format(top5_prob[i].item() * 100))
