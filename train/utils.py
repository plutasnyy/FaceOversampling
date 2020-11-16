def show_batch_of_images(data_module):
    import matplotlib.pyplot as plt

    def denormalise(image):
        image = image.numpy().transpose(1, 2, 0)  # PIL images have channel last
        mean = [0.485, 0.456, 0.406]
        stdd = [0.229, 0.224, 0.225]
        image = (image * stdd + mean).clip(0, 1)
        return image

    example_rows = 4
    example_cols = 8

    images, y = next(iter(data_module.train_dataloader()))
    plt.rcParams['figure.dpi'] = 120  # Increase size of pyplot plots

    fig, axes = plt.subplots(example_rows, example_cols, figsize=(9, 5))  # sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, image, label in zip(axes, images, y):
        ax.imshow(denormalise(image))
        ax.set_axis_off()
        ax.set_title(int(label), fontsize=7)

    fig.subplots_adjust(wspace=0.02, hspace=0)
    fig.suptitle('Augmented training set images', fontsize=20)
    plt.show()
