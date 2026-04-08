#! usr/bin/python3
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from dataset import ConveyorSimulator
from metrics import AccuracyMetric, MeanAveragePrecisionMetric, SegmentationIntersectionOverUnionMetric
from visualizer import Visualizer

from models.classification_network import build_classification_model
from models.detection_network import build_detection_model
from models.segmentation_network import build_segmentation_model

TRAIN_VALIDATION_SPLIT = 0.9
CLASS_PROBABILITY_THRESHOLD = 0.5
INTERSECTION_OVER_UNION_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
SEGMENTATION_BACKGROUND_CLASS = 3


class ConveyorCnnTrainer():
    def __init__(self, args):
        self._args = args
        # Initialisation de pytorch
        # Always use GPU if available (ignore use_gpu flag)
        use_cuda = torch.cuda.is_available()
        self._device = torch.device('cuda' if use_cuda else 'cpu')
        if use_cuda:
            print(f'✅ GPU Enabled: {torch.cuda.get_device_name(0)}')
        seed = np.random.rand()
        torch.manual_seed(seed)
        
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Generation des 'path'
        self._dir_path = os.path.dirname(__file__)
        self._train_data_path = os.path.join(self._dir_path, 'data', 'training')
        self._test_data_path = os.path.join(self._dir_path, 'data', 'test')
        self._weights_path = os.path.join(self._dir_path, 'weights', 'task_' + self._args.task + '_best.pt')
        self._history_path = os.path.join(self._dir_path, 'weights', 'task_' + self._args.task + '_history.json')

        # Initialisation du model et des classes pour l'entraînement
        self._model = self._create_model(self._args.task).to(self._device)
        self._criterion = self._create_criterion(self._args.task)

        print('Model : ')
        print(self._model)
        print('\nNumber of parameters in the model : ', sum(p.numel() for p in self._model.parameters()))

    def _create_model(self, task):
        if task == 'classification':
            return build_classification_model()
        elif task == 'detection':
            return build_detection_model()
        elif task == 'segmentation':
            return build_segmentation_model()
        else:
            raise ValueError('Not supported task')

    def _create_criterion(self, task):
        if task == 'classification':
            return torch.nn.CrossEntropyLoss()
        elif task == 'detection':
            return torch.nn.MSELoss()
        elif task == 'segmentation':
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('Not supported task')

    def _create_metric(self, task):
        if task == 'classification':
            return AccuracyMetric(CLASS_PROBABILITY_THRESHOLD)
        elif task == 'detection':
            return MeanAveragePrecisionMetric(3, INTERSECTION_OVER_UNION_THRESHOLD)
        elif task == 'segmentation':
            return SegmentationIntersectionOverUnionMetric(SEGMENTATION_BACKGROUND_CLASS)
        else:
            raise ValueError('Not supported task')

    def test(self):
        params_test = {'batch_size': self._args.batch_size, 'shuffle': False, 'num_workers': 4}

        dataset_test = ConveyorSimulator(self._test_data_path, self.transform)
        test_loader = torch.utils.data.DataLoader(dataset_test, **params_test)

        test_metric = self._create_metric(self._args.task)
        visualizer = Visualizer('test', self._args.task, CLASS_PROBABILITY_THRESHOLD, CONFIDENCE_THRESHOLD,
                                SEGMENTATION_BACKGROUND_CLASS)

        print('Test data : ', len(dataset_test))
        self._model.load_state_dict(torch.load(self._weights_path))
        self._model.eval()

        test_loss = 0
        with torch.no_grad():
            for image, segmentation_target, boxes, class_labels in test_loader:
                image = image.to(self._device)
                segmentation_target = segmentation_target.to(self._device)
                boxes = boxes.to(self._device)
                class_labels = class_labels.to(self._device)

                loss = self._test_batch(self._args.task, self._model, self._criterion, test_metric,
                                        image, segmentation_target, boxes, class_labels)
                test_loss += loss.item()

        test_loss /= len(dataset_test)
        print('Test - Average loss: {:.6f}, {}: {:.6f}'.format(
            test_loss, test_metric.get_name(), test_metric.get_value()))

        prediction = self._model(image)
        visualizer.show_prediction(image[0], prediction[0], segmentation_target[0], boxes[0], class_labels[0])

    def train(self):
        epochs_train_losses = []
        epochs_validation_losses = []
        epochs_train_metrics = []
        epochs_validation_metrics = []
        best_validation = 0
        nb_worse_validation = 0

        if self._args.use_weights and os.path.exists(self._weights_path):
            print(f'Resuming from {self._weights_path}')
            self._model.load_state_dict(torch.load(self._weights_path, map_location=self._device))
            if os.path.exists(self._history_path):
                with open(self._history_path) as f:
                    history = json.load(f)
                epochs_train_losses = history['train_losses']
                epochs_validation_losses = history['val_losses']
                epochs_train_metrics = history['train_metrics']
                epochs_validation_metrics = history['val_metrics']
                best_validation = max(epochs_validation_metrics)
                print(f'Resuming history from epoch {len(epochs_train_losses)} (best val metric: {best_validation:.4f})')

        params_train = {'batch_size': self._args.batch_size, 'shuffle': True, 'num_workers': 4}
        params_validation = {'batch_size': self._args.batch_size, 'shuffle': False, 'num_workers': 4}

        dataset_trainval = ConveyorSimulator(self._train_data_path, self.transform)
        dataset_train, dataset_validation = torch.utils.data.random_split(dataset_trainval,
                                                                          [int(len(
                                                                              dataset_trainval) * TRAIN_VALIDATION_SPLIT),
                                                                           int(len(dataset_trainval) - int(len(
                                                                               dataset_trainval) * TRAIN_VALIDATION_SPLIT))])
        train_loader = torch.utils.data.DataLoader(dataset_train, **params_train)
        validation_loader = torch.utils.data.DataLoader(dataset_validation, **params_validation)

        print('Number of epochs : ', self._args.epochs)
        print('Training data : ', len(dataset_train))
        print('Validation data : ', len(dataset_validation))

        optimizer = optim.Adam(self._model.parameters(), lr=self._args.lr, weight_decay=1e-4)
        train_metric = self._create_metric(self._args.task)
        validation_metric = self._create_metric(self._args.task)

        visualizer = Visualizer('train', self._args.task, CLASS_PROBABILITY_THRESHOLD, CONFIDENCE_THRESHOLD,
                                SEGMENTATION_BACKGROUND_CLASS)

        for epoch in range(1, self._args.epochs + 1):
            print('\nEpoch: {}'.format(epoch))
            # Entraînement
            self._model.train()
            train_loss = 0
            train_metric.clear()

            # Boucle pour chaque batch
            for image, segmentation_target, boxes, class_labels in train_loader:
                image = image.to(self._device)
                segmentation_target = segmentation_target.to(self._device)
                boxes = boxes.to(self._device)
                class_labels = class_labels.to(self._device)

                loss = self._train_batch(self._args.task, self._model, self._criterion, train_metric, optimizer,
                                         image, segmentation_target, boxes, class_labels)

                train_loss += loss.item()

            # Affichage après la batch
            train_loss = train_loss / len(dataset_train)
            epochs_train_losses.append(train_loss)
            epochs_train_metrics.append(train_metric.get_value())
            print('Train - Average Loss: {:.6f}, {}: {:.6f}'.format(
                train_loss, train_metric.get_name(), train_metric.get_value()))

            # Validation
            self._model.eval()
            validation_loss = 0
            validation_metric.clear()
            with torch.no_grad():
                for image, masks, boxes, labels in validation_loader:
                    image = image.to(self._device)
                    masks = masks.to(self._device)
                    boxes = boxes.to(self._device)
                    labels = labels.to(self._device)

                    loss = self._test_batch(self._args.task, self._model, self._criterion, validation_metric,
                                            image, masks, boxes, labels)
                    validation_loss += loss.item()

            validation_metric_value = validation_metric.get_value()
            validation_loss /= len(dataset_validation)
            if validation_metric_value > best_validation:
                best_validation = validation_metric_value
                nb_worse_validation = 0
                print('Saving new best model')
                torch.save(self._model.state_dict(), self._weights_path)
            else:
                nb_worse_validation += 1

            epochs_validation_losses.append(validation_loss)
            epochs_validation_metrics.append(validation_metric.get_value())
            print('Validation - Average loss: {:.6f}, {}: {:.6f}'.format(
                validation_loss, validation_metric.get_name(), validation_metric_value))

            prediction = self._model(image)
            visualizer.show_prediction(image[0], prediction[0], masks[0], boxes[0], labels[0])
            visualizer.show_learning_curves(epochs_train_losses, epochs_validation_losses,
                                            epochs_train_metrics, epochs_validation_metrics,
                                            train_metric.get_name())
            with open(self._history_path, 'w') as f:
                json.dump({
                    'train_losses': epochs_train_losses,
                    'val_losses': epochs_validation_losses,
                    'train_metrics': epochs_train_metrics,
                    'val_metrics': epochs_validation_metrics,
                }, f)

        ans = input('Do you want ot test? (y/n):')
        if ans == 'y':
            self.test()

    def _boxes_to_target7(self, boxes):
        """
        Converts detection target from (N, 3, 5) to (N, 3, 7) for MSELoss with the detection network.

        Target format (N, 3, 5):  [confidence, x, y, size, class_index]
        Output format (N, 3, 7):  [confidence, x, y, size, score_circle, score_triangle, score_cross]

        The class index is one-hot encoded into 3 class scores.
        For empty slots (confidence=0), class scores stay at 0.
        """
        N, M, _ = boxes.shape
        target7 = torch.zeros(N, M, 7, device=boxes.device, dtype=boxes.dtype)
        target7[:, :, :4] = boxes[:, :, :4]
        present = (boxes[:, :, 0] == 1)
        cls_idx = boxes[:, :, 4].long().clamp(0, 2)
        one_hot = torch.zeros(N, M, 3, device=boxes.device, dtype=boxes.dtype)
        one_hot.scatter_(2, cls_idx.unsqueeze(2), 1.0)
        target7[:, :, 4:] = one_hot * present.unsqueeze(2)
        return target7

    def _detection_loss(self, prediction, target):
        """
        YOLO-style weighted detection loss.

        Uses reduction='none' + explicit masking so empty slots contribute
        exactly zero gradient to coord and class terms (multiplying logits by
        obj_mask before BCE gives BCE(0,0)=log(2)≠0, which was poisoning training).

        Weights:
          - λ_coord=5  : bbox regression, normalized by number of objects
          - λ_noobj=0.5: confidence on empty slots
          - class loss normalized by number of objects
        """
        obj_mask = target[..., 0:1]        # (N, 3, 1) — 1 where object present
        noobj_mask = 1.0 - obj_mask
        n_obj = obj_mask.sum().clamp(min=1)
        n_noobj = noobj_mask.sum().clamp(min=1)

        # Coordinate loss — reduction='none', then zero out empty slots, then mean over objects
        coord_loss = (
            F.smooth_l1_loss(prediction[..., 1:4], target[..., 1:4], reduction='none')
            * obj_mask
        ).sum() / n_obj

        # Confidence loss — compute per-element, then split by mask
        raw_conf = F.binary_cross_entropy_with_logits(
            prediction[..., 0:1], target[..., 0:1], reduction='none')
        obj_conf_loss   = (raw_conf * obj_mask).sum()   / n_obj
        noobj_conf_loss = (raw_conf * noobj_mask).sum() / n_noobj

        # Class loss — reduction='none', zero out empty slots, mean over objects
        class_loss = (
            F.binary_cross_entropy_with_logits(prediction[..., 4:], target[..., 4:], reduction='none')
            * obj_mask
        ).sum() / n_obj

        return 5.0 * coord_loss + obj_conf_loss + 0.5 * noobj_conf_loss + class_loss

    def _train_batch(self, task, model, criterion, metric, optimizer, image, segmentation_target, boxes, class_labels):
        """
        Méthode qui effectue une passe d'entraînement sur un lot de données.
        Vous devez appeler la méthode "accumulate" de l'objet "metric" pour que le calcul de la métrique se fasse.
        La définition des paramètres de cette méthode se trouve dans le fichier "metrics.py"
        N: La taille du lot (batch size)
        H: La hauteur des images
        W: La largeur des images

        :param task: La tâche à effectuer ('classification', 'detection' ou 'segmentation')
        :param model: Le modèle créé dans create_model
        :param criterion: La fonction de coût créée dans create_criterion
        :param metric: La métrique créée dans create_metric
        :param optimizer: L'optimisateur pour entraîner le modèle
        :param image: Le tenseur contenant les images du lot à passer au modèle
            Dimensions : (N, 1, H, W)
        :param segmentation_target: Le tenseur cible pour la tâche de segmentation qui contient l'indice de la classe pour chaque pixel
            Dimensions : (N, H, W)
        :param boxes: Le tenseur cible pour la tâche de détection:
            Dimensions: (N, 3, 5)
                Si un 1 est présent à (i, j, 0), le vecteur (i, j, 0:5) représente un objet.
                Si un 0 est présent à (i, j, 0), le vecteur (i, j, 0:5) ne représente aucun objet.
                Si le vecteur représente un objet (i, j, :):
                    (i, j, 1) est la position x centrale normalisée de l'objet j de l'image i.
                    (i, j, 2) est la position y centrale normalisée de l'objet j de l'image i.
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet j de l'image i.
                    (i, j, 4) est l'indice de la classe de l'objet j de l'image i.

        :param class_labels: Le tenseur cible pour la tâche de classification
            Dimensions : (N, 3)
                Si un 1 est présent à (i, 0), un cercle est présent dans l'image i.
                Si un 0 est présent à (i, 0), aucun cercle n'est présent dans l'image i.
                Si un 1 est présent à (i, 1), un triangle est présent dans l'image i.
                Si un 0 est présent à (i, 1), aucun triangle n'est présent dans l'image i.
                Si un 1 est présent à (i, 2), une croix est présente dans l'image i.
                Si un 0 est présent à (i, 2), aucune croix n'est présente dans l'image i.
        :return: La valeur de la fonction de coût pour le lot
        """

        optimizer.zero_grad()
        prediction = model(image)
        if task == 'classification':
            target = class_labels
        elif task == 'detection':
            target = self._boxes_to_target7(boxes)
        elif task == 'segmentation':
            target = segmentation_target
        else:
            raise ValueError('Not supported task')
        metric.accumulate(prediction, boxes if task == 'detection' else target)
        if task == 'detection':
            loss = self._detection_loss(prediction, target)
        else:
            loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        return loss


    def _test_batch(self, task, model, criterion, metric, image, segmentation_target, boxes, class_labels):
        """
        Méthode qui effectue une passe de validation ou de test sur un lot de données.
        Vous devez appeler la méthode "accumulate" de l'objet "metric" pour que le calcul de la métrique se fasse.
        La définition des paramètres de cette méthode se trouve dans le fichier "metrics.py"
        N: La taille du lot (batch size)
        H: La hauteur des images
        W: La largeur des images

        :param task: La tâche à effectuer ('classification', 'detection' ou 'segmentation')
        :param model: Le modèle créé dans create_model
        :param criterion: La fonction de coût créée dans create_criterion
        :param metric: La métrique créée dans create_metric
        :param image: Le tenseur PyTorch contenant les images du lot à passer au modèle
            Dimensions : (N, 1, H, W)
        :param segmentation_target: Le tenseur PyTorch cible pour la tâche de segmentation qui contient l'indice de la classe pour chaque pixel
            Dimensions : (N, H, W)
        :param boxes: Le tenseur PyTorch cible pour la tâche de détection:
            Dimensions: (N, 3, 5)
                Si un 1 est présent à (i, j, 0), le vecteur (i, j, 0:5) représente un objet.
                Si un 0 est présent à (i, j, 0), le vecteur (i, j, 0:5) ne représente aucun objet.
                Si le vecteur représente un objet (i, j, :):
                    (i, j, 1) est la position x centrale normalisée de l'objet j de l'image i.
                    (i, j, 2) est la position y centrale normalisée de l'objet j de l'image i.
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet j de l'image i.
                    (i, j, 4) est l'indice de la classe de l'objet j de l'image i.

        :param class_labels: Le tenseur PyTorch cible pour la tâche de classification
            Dimensions : (N, 3)
                Si un 1 est présent à (i, 0), un cercle est présent dans l'image i.
                Si un 0 est présent à (i, 0), aucun cercle n'est présent dans l'image i.
                Si un 1 est présent à (i, 1), un triangle est présent dans l'image i.
                Si un 0 est présent à (i, 1), aucun triangle n'est présent dans l'image i.
                Si un 1 est présent à (i, 2), une croix est présente dans l'image i.
                Si un 0 est présent à (i, 2), aucune croix n'est présente dans l'image i.
        :return: La valeur de la fonction de coût pour le lot
        """

        prediction = model(image)
        if task == 'classification':
            target = class_labels
        elif task == 'detection':
            target = self._boxes_to_target7(boxes)
        elif task == 'segmentation':
            target = segmentation_target
        else:
            raise ValueError('Not supported task')
        if task == 'detection':
            loss = self._detection_loss(prediction, target)
        else:
            loss = criterion(prediction, target)
        metric.accumulate(prediction, boxes if task == 'detection' else target)
        return loss

if __name__ == '__main__':
    #  Settings
    parser = argparse.ArgumentParser(description='Conveyor CNN')
    parser.add_argument('--mode', choices=['train', 'test'], help='The script mode', required=True)
    parser.add_argument('--task', choices=['classification', 'detection', 'segmentation'],
                        help='The CNN task', required=True)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for training (default: 20)')
    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate used for training (default: 4e-4)')
    parser.add_argument('--use_gpu', action='store_true', help='use the gpu instead of the cpu')
    parser.add_argument('--early_stop', type=int, default=25,
                        help='number of worse validation loss before quitting training (default: 25)')
    parser.add_argument('--use_weights', action='store_true',
                        help='resume training from existing saved weights and history')

    args = parser.parse_args()

    conv = ConveyorCnnTrainer(args)

    if args.mode == 'train':
        print('\n--- Training mode ---\n')
        conv.train()
    elif args.mode == 'test':
        print('\n--- Testing mode ---\n')
        conv.test()
