import copy
import numpy as np
from numba.cuda import is_available
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn


class inference:
    def __init__(self, cudaId, dataset, model, pthPath):
        self.val_loader = dataset
        self.model = model
        self.device = torch.device(f"cuda:{cudaId}" if is_available() else "cpu")
        self.examinData_batchSize = 32

        model_state_dict = torch.load(pthPath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_state_dict)

        self.criterion = nn.CrossEntropyLoss()

        print(f"Examin device online")
        print(f'{self.device} available')

    def examin(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        acc = 0
        count = 0
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)

                targets = targets.long().to(self.device)  # CE
                outputs = self.model(inputs)

                npOutputs = torch.argmax(outputs, dim=1)
                npTargets = targets  # CE

                npOutputs = np.array(npOutputs.cpu())
                npTargets = np.array(npTargets.cpu())

                for i in range(len(npOutputs)):
                    singleOutput = npOutputs[i]
                    singleTarget = npTargets[i]

                    count += 1
                    if singleOutput == singleTarget:
                        acc += 1

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            avg_loss = total_loss / len(self.val_loader)
            acc /= count
            acc *= 100
            print(f"Server validation Loss: {avg_loss:.4f} | accuracy: {acc: .4f}")

        return avg_loss, acc, all_targets, all_outputs

    def __del__(self):
        print('examiner offline')