import torch
import torch.nn as nn


class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=90, dropout_rate=0.3):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model

        # Logit scaling for improved stability in classification
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # Classifier with enhanced regularization and modularity
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Initialize weights for better stability
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, images):
        # Extract image features from the CLIP model
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.float()

        # Pass features through the classifier
        logits = self.classifier(image_features)
        return logits