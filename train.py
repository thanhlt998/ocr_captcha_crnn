from crnn import *


# config
class Config:
    # data
    train_data_path = Path('data/images')
    # train_data_path = Path('data/kaggle_data/samples')
    test_data_path = Path('data/test_images')
    model_dir = Path('models')

    # model architecture
    in_channels = 3
    rnn_hidden_size = 256
    leaky_relu = False

    # training
    n_epochs = 5
    lr = 1e-3
    bs = 64
    model_name = 'ocr_crnn_captcha'


def get_ds(items):
    item_tfms = [PILImage.create, ]
    y_tfms = [label_func, CategorizeList(add_na=False), ]

    ds = Datasets(
        items=items,
        tfms=[
            item_tfms,
            y_tfms,
        ],
        splits=RandomSplitter(valid_pct=0.2, seed=42)(items),
    )
    return ds


def get_dls(ds, bs=64):
    dls = ds.dataloaders(
        bs=bs,
        before_batch=BeforeBatchTransform(keep_ratio=True),
        create_batch=CreateBatchTransform(),
        after_batch=[IntToFloatTensor, Normalize.from_stats([0.5] * 3, [0.5] * 3)],
    )
    return dls


def train(config):
    # load data
    f_names = get_image_files(config.train_data_path)
    ds = get_ds(items=f_names)
    dls = get_dls(ds, bs=config.bs)

    # create model
    model = CRNN(
        in_channels=config.in_channels,
        rnn_hidden_size=config.rnn_hidden_size,
        n_classes=ds.tfms[1][-1].n_classes,
        leaky_relu=config.leaky_relu,
    )
    loss_func = CTCLoss(blank=ds.tfms[1][-1].blank_idx)
    metrics = [AccMetric()]

    # create learner
    learner = Learner(
        dls=dls,
        model=model,
        loss_func=loss_func,
        metrics=metrics,
    )

    # fit one cycle
    learner.fit_one_cycle(config.n_epochs, lr=config.lr)

    learner.model_dir = config.model_dir
    learner.save(config.model_name)

    learner.export(config.model_dir / f'{config.model_name}.pkl')


def evaluate(config):
    # load learner
    learner = load_learner(config.model_dir / f'{config.model_name}.pkl')

    # create test_dl
    test_files = get_image_files(config.test_data_path)
    test_dl = learner.dls.test_dl(test_files, with_labels=True)

    # validate test_dl
    test_loss, test_acc = learner.validate(dl=test_dl)
    print(f'test_loss = {test_loss}, test_acc = {test_acc}')


if __name__ == '__main__':
    config = Config()
    print('-' * 10, 'Training', '-' * 10)
    train(config)

    print('-' * 30)

    print('-' * 10, 'Test', '-' * 10)
    evaluate(config)
