from crnn import *


def get_torch_script(export_learner_path, save_script_path):
    learner = load_learner(export_learner_path)
    one_batch = torch.rand(1, 3, 32, 160)

    script_module = torch.jit.trace(learner.model, example_inputs=one_batch)
    script_module.save(save_script_path)


def model2script(export_learner_path, save_script_path):
    learner = load_learner(export_learner_path)
    f = learner.model.eval()
    types_to_quantize = {nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LSTM}
    q = torch.quantization.quantize_dynamic(f, types_to_quantize, dtype=torch.qint8)
    script = torch.jit.script(q)
    torch.jit.save(script, save_script_path)


def dump_dls_empty(export_learner_path, save_dls_path):
    learner = load_learner(export_learner_path)
    torch.save(learner.dls, save_dls_path)


def remove_loss_func_from_learner(export_learner_path, save_path):
    learner = load_learner(export_learner_path)
    learner.loss_func = None
    learner.export(save_path)


if __name__ == '__main__':
    export_path = 'models/captcha_6271_data6.pkl'
    # save_path = 'models/captcha_script_module.pt'
    # save_path = 'model_serving/quantized_captcha_script.pt'
    # save_dls_path = 'model_serving/dls_empty.pt'
    save_path_new_export = 'model_serving/captcha_data6.pkl'

    # get_torch_script(export_learner_path=export_path, save_script_path=save_path)
    # model2script(export_learner_path=export_path, save_script_path=save_path)
    # dump_dls_empty(export_learner_path=export_path, save_dls_path=save_dls_path)
    remove_loss_func_from_learner(export_learner_path=export_path, save_path=save_path_new_export)