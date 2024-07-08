from Mnist import DenoisingGAN
# Load the dataset
train_dataset = ...

# Create a DenoisingGAN instance
model = DenoisingGAN((3, 32, 32), (3, 32, 32), num_layers=2)

# Train the model
model.train(train_dataset, epochs=10, batch_size=32, lr=0.001)