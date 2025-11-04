from torch import nn
import torch
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, n_input_features, n_channels, n_reduces):
        super().__init__()
        self._1x1_conv = nn.Conv2d(
            n_input_features, n_channels["1x1"], kernel_size=1, stride=1, padding="same"
        )

        self._3x3_reducer = nn.Conv2d(
            n_input_features, n_reduces["3x3"], kernel_size=1, stride=1, padding="same"
        )
        self._3x3_conv = nn.Conv2d(
            n_reduces["3x3"], n_channels["3x3"], kernel_size=3, stride=1, padding="same"
        )

        self._5x5_reducer = nn.Conv2d(
            n_input_features, n_reduces["5x5"], kernel_size=1, stride=1, padding="same"
        )
        self._5x5_conv = nn.Conv2d(
            n_reduces["5x5"], n_channels["5x5"], kernel_size=5, stride=1, padding="same"
        )

        self._pooler = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self._pooler_1x1 = nn.Conv2d(
            n_input_features,
            n_channels["maxpool"],
            kernel_size=1,
            stride=1,
            padding="same",
        )

    def forward(self, input_tensor):
        conv_1x1_result = F.relu(self._1x1_conv(input_tensor))

        conv_3x3_result = F.relu(
            self._3x3_conv(F.relu(self._3x3_reducer(input_tensor)))
        )

        conv_5x5_result = F.relu(
            self._5x5_conv(F.relu(self._5x5_reducer(input_tensor)))
        )

        pooler_result = F.relu(self._pooler_1x1(F.relu(self._pooler(input_tensor))))

        return torch.concat(
            [conv_1x1_result, conv_3x3_result, conv_5x5_result, pooler_result], dim=1
        )


inception_module = InceptionModule(
    192, {"1x1": 64, "3x3": 128, "5x5": 32, "maxpool": 32}, {"3x3": 96, "5x5": 16}
)

input_tensor = torch.randn((32, 192, 28, 28), dtype=torch.float32)
output_tensor = inception_module(input_tensor)
print(output_tensor.shape)

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, 3, 1, "same"),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(3, 2, padding=1),
    InceptionModule(
        32, {"1x1": 16, "3x3": 32, "5x5": 8, "maxpool": 8}, {"3x3": 16, "5x5": 4}
    ),
    InceptionModule(
        64, {"1x1": 32, "3x3": 64, "5x5": 16, "maxpool": 16}, {"3x3": 32, "5x5": 8}
    ),
    torch.nn.Conv2d(128, 128, 3, 1, "same"),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(3, 2, padding=1),
    InceptionModule(
        128, {"1x1": 32, "3x3": 64, "5x5": 16, "maxpool": 16}, {"3x3": 32, "5x5": 8}
    ),
    InceptionModule(
        128, {"1x1": 32, "3x3": 64, "5x5": 16, "maxpool": 16}, {"3x3": 32, "5x5": 8}
    ),
    torch.nn.Conv2d(128, 128, 3, 1, "same"),
    torch.nn.ReLU(),
    torch.nn.AvgPool2d(7),
    torch.nn.Flatten(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10),
    torch.nn.Softmax(dim=1),
)

dummy_image = torch.randn((32, 1, 28, 28), dtype=torch.float32)

output = model(dummy_image)

print(output.shape)
# Output
# torch.Size([32, 10])
