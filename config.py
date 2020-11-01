batch_size = 64
num_classes = 7
num_layers = 1
num_features = 84
learning_rate = 0.001
num_epochs = 20
hidden_dimension = 32
num_steps = 128
sampling_rate = 128
validation_split = 0.1
early_stop_threshold = 0.0005
weighted_loss = True
save_plot = True

categories = [('Bluetooth', 1), ('CPU', 2), ('Display', 3), ('Location', 4), ('Network', 5), ('Sensor', 6)]
subcategories = [('B1', 1), ('B2',2), ('B3', 3), ('C1', 4), ('C2', 5), ('C3', 6), ('C4', 7), ('D1', 8), ('D2', 9),
                 ('L1', 10), ('L2', 11), ('L3', 12), ('L4', 13), ('N1', 14), ('N2', 15), ('N3', 16), ('N4', 17),
                 ('N5', 18), ('N6', 19), ('N7', 20), ('S1', 21), ('S2', 22)]
gamma = 0.8