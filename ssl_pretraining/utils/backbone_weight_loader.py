import torch

def load_backbone_weights(model, pretrained_backbone_path, load_pretrained, model_name = None):
    checkpoint = torch.load(pretrained_backbone_path, map_location='cpu')
    if 'model_state_dict' in checkpoint.keys():
        checkpoint = checkpoint['model_state_dict']

    model_dict = model.state_dict()
    checkpoint_dict = checkpoint
    checkpoint_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint_dict.items()}

    new_state_dict = {}
    removed_keys = ['norm.weight', 'norm.bias', 'norm.running_mean', 'norm.running_var', 'norm.num_batches_tracked', 'head.weight', 'head.bias']

    if model_name == "vqvae_codebook":

        for key, value in checkpoint_dict.items():
            if 'patch_embed1.proj.bias' not in key:

                if key.startswith("student."):
                    new_key = "encoder." + key[8:]
                else:
                    new_key = key

                new_state_dict[new_key] = value

            else:
                if key.startswith("student."):
                    new_key = key[8:]
                else:
                    new_key = key

                new_state_dict[new_key] = value

    elif model_name == "OpticalPromptedUniFormer":
        for key, value in checkpoint_dict.items():
            new_key = key[8:] if key.startswith("student.") else key
            new_state_dict[new_key] = value
    elif model_name == "PCRL":
        for key, value in checkpoint_dict["state_dict"].items():
            new_state_dict[key] = value
    else:
        for key, value in checkpoint_dict.items():
            new_state_dict[key] = value

    model_dict.update(new_state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

    print("-----------------------------------------------------------------------")

    if missing_keys:
        print("Missing keys:")
        for key in missing_keys:
            print(key)

    if unexpected_keys:
        print("Unexpected keys:")
        for key in unexpected_keys:
            print(key)
    print("Backbone weights loaded successfully!")

    return model
