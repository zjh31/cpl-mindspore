import mindspore as ms

import torch
import argparse
# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')['model']
    pt_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        if name == 'featpool.conv.weight':
            parameter = parameter.unsqueeze(3)
        if 'output.dense.weight' in name or 'intermediate.dense.weight' in name:
            parameter = parameter.t()
        print(name, parameter.numpy().shape)
        pt_params[name] = parameter.numpy()
    return pt_params

# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params




def param_convert(ms_params, pt_params, ckpt_path):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
            "beta": "bias",
            "moving_mean": "running_mean",
            "moving_variance": "running_var"}
    
    """

textmodel.bert.encoder.layer.0.attention.output.dense.weight (768, 768)
textmodel.bert.encoder.layer.0.attention.output.dense.bias (768,)






textmodel.bert.bert_encoder.encoder.blocks.0.attention.projection.weight (768, 768)
textmodel.bert.bert_encoder.encoder.blocks.0.attention.projection.bias (768,)



    """
    
    ms2pt = {"textmodel.bert.word_embedding.embedding_table": "textmodel.bert.embeddings.word_embeddings.weight",
                "textmodel.bert.embedding_postprocessor.token_type_embedding.embedding_table": "textmodel.bert.embeddings.token_type_embeddings.weight",
                "textmodel.bert.embedding_postprocessor.full_position_embedding.embedding_table": "textmodel.bert.embeddings.position_embeddings.weight",
                "textmodel.bert.embedding_postprocessor.layernorm.gamma": "textmodel.bert.embeddings.LayerNorm.weight",
                "textmodel.bert.embedding_postprocessor.layernorm.beta": "textmodel.bert.embeddings.LayerNorm.bias",
                "textmodel.bert.dense.weight": "textmodel.bert.pooler.dense.weight",
                "textmodel.bert.dense.bias": "textmodel.bert.pooler.dense.bias",
                #"textmodel.bert.layernorm.gamma": "textmodel.layernorm.weight",
                #"textmodel.bert.layernorm.beta": "textmodel.layernorm.bias",
                }
    layer_map = {# layer0
                "textmodel.bert.bert_encoder.encoder.blocks.#.layernorm1.gamma": "textmodel.bert.encoder.layer.#.output.LayerNorm.weight",
                "textmodel.bert.bert_encoder.encoder.blocks.#.layernorm1.beta": "textmodel.bert.encoder.layer.#.output.LayerNorm.bias",
                "textmodel.bert.bert_encoder.encoder.blocks.#.layernorm2.gamma": "textmodel.bert.encoder.layer.#.attention.output.LayerNorm.weight",
                "textmodel.bert.bert_encoder.encoder.blocks.#.layernorm2.beta": "textmodel.bert.encoder.layer.#.attention.output.LayerNorm.bias",
                "textmodel.bert.bert_encoder.encoder.blocks.#.attention.dense1.weight": "textmodel.bert.encoder.layer.#.attention.self.query.weight",
                "textmodel.bert.bert_encoder.encoder.blocks.#.attention.dense1.bias": "textmodel.bert.encoder.layer.#.attention.self.query.bias",
                "textmodel.bert.bert_encoder.encoder.blocks.#.attention.dense2.weight": "textmodel.bert.encoder.layer.#.attention.self.key.weight",
                "textmodel.bert.bert_encoder.encoder.blocks.#.attention.dense2.bias": "textmodel.bert.encoder.layer.#.attention.self.key.bias",
                "textmodel.bert.bert_encoder.encoder.blocks.#.attention.dense3.weight": "textmodel.bert.encoder.layer.#.attention.self.value.weight",
                "textmodel.bert.bert_encoder.encoder.blocks.#.attention.dense3.bias": "textmodel.bert.encoder.layer.#.attention.self.value.bias",
                "textmodel.bert.bert_encoder.encoder.blocks.#.output.mapping.weight": "textmodel.bert.encoder.layer.#.intermediate.dense.weight",
                "textmodel.bert.bert_encoder.encoder.blocks.#.output.mapping.bias": "textmodel.bert.encoder.layer.#.intermediate.dense.bias",
                "textmodel.bert.bert_encoder.encoder.blocks.#.output.projection.weight": "textmodel.bert.encoder.layer.#.output.dense.weight",
                "textmodel.bert.bert_encoder.encoder.blocks.#.output.projection.bias": "textmodel.bert.encoder.layer.#.output.dense.bias",
                "textmodel.bert.bert_encoder.encoder.blocks.#.attention.projection.weight": "textmodel.bert.encoder.layer.#.attention.output.dense.weight",
                "textmodel.bert.bert_encoder.encoder.blocks.#.attention.projection.bias": "textmodel.bert.encoder.layer.#.attention.output.dense.bias"}
    for i in range(12):
        for k, v in layer_map.items():
            ms2pt[k.replace('#', str(i))] = v.replace('#', str(i))
    new_params_list = []
    for ms_param in ms_params.keys():
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数
        if "bn" in ms_param or "downsample.1" in ms_param:
            ms_param_item = ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表

            if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            elif ms2pt.get(ms_param,'') in pt_params and pt_params[ms2pt[ms_param]].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms2pt[ms_param]]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    print(len(new_params_list))
    ms.save_checkpoint(new_params_list, ckpt_path)



def main():
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument(
        "--config-file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_model(cfg)
    pth_path = "/hd1/shared/TRM_pytorch/outputs/activitynet/pool_model_5e.pth"
    pt_param = pytorch_params(pth_path)
    print("="*20)
    ckpt_path = "trm_act_e5.ckpt"
    ms_param = mindspore_params(model)
    param_convert(ms_param, pt_param, ckpt_path)

if __name__ == "__main__":
    #mp.set_start_method('spawn')
    #
    main()