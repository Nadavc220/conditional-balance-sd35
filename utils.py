
def generate_file_name(num_style_layers, controlnet_removal_amount, content_image_path, seed):
    starter = content_image_path.split('/')[-1].split('.')[0] if content_image_path else 'gen'
    addition = f"s{num_style_layers}"

    if controlnet_removal_amount:
        addition = addition + f"_t{controlnet_removal_amount}"

    f_name = f"{starter}_{addition}_{seed}.png"
    return f_name