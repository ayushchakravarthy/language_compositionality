import json
import pickle
import matplotlib.pyplot as plt
import numpy as np

from einops import rearrange


def plot(run, args):
    attn_file = f"results/{args.results_dir}/{args.out_attn_wts}{run}.pickle"
    stats_file = f"results/{args.results_dir}/{args.out_data_file}{run}.json"
    
    with open(attn_file, 'rb') as f:
        attn_maps = pickle.load(f)
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Compute mean and std-dev on the testing set
    test_accs = stats.get('test_accs')
    mean = sum(test_accs) / len(test_accs)
    std = sum([((x - mean) ** 2) for x in test_accs]) / len(test_accs)
    res = std ** 0.5
    print(f"Test accuracy: {mean} +- {res}")

    # Plot loss curve
    plt.plot(test_accs)
    plt.xlabel("Epochs")
    plt.ylabel("Testing accuracy")

    plt.show()
    plt.savefig(f"{args.model_type}_test_results.png")

    # Plot attention maps


if __name__ == "__main__":
    attn_file = 'results/language_parser/train_defaults_jump_attn_maps0.pickle'
    res_file = 'results/language_parser/train_defaults_jump0.json'

    with open(attn_file, 'rb') as f:
        attn_maps = pickle.load(f)

    with open(res_file, 'r') as f:
        stats = json.load(f)

    test_accs = stats.get('test_accs')
    mean = sum(test_accs) / len(test_accs)
    std = sum([((x - mean) ** 2) for x in test_accs]) / len(test_accs)
    res = std ** 0.5
    print(f"Test accuracy: {mean} +- {res}")
    # plt.plot(stats.get('test_accs'))
    # plt.show()
    # plt.savefig('transformer_test_results.png')

#     for i in range(num_blocks):
#         block_attn_maps = encoder_maps[i]


    encoder_maps = attn_maps.get('Encoder')
    decoder_maps = attn_maps.get('Decoder')

    block_0_attn_maps = encoder_maps[0]
    enc_0_attn_maps = block_0_attn_maps[0]
    enc_0_attn_maps = enc_0_attn_maps[-1].mean(axis=1)
    plt.imsave("enc_0_attn_map.png", enc_0_attn_maps)
    
    dec_0_1_attn_map = block_0_attn_maps[1][0][-1].mean(axis=1).T
    plt.imsave("dec_0_1_attn_map.png", dec_0_1_attn_map)
    dec_0_2_attn_map = block_0_attn_maps[1][1][-1].mean(axis=1).T
    plt.imsave("dec_0_2_attn_map.png", dec_0_2_attn_map)

    block_1_attn_maps = encoder_maps[1]
    enc_1_attn_maps = block_1_attn_maps[0][-1].mean(axis=1)
    plt.imsave("enc_1_attn_map.png", enc_1_attn_maps)
    
    dec_1_1_attn_map = block_1_attn_maps[1][0][-1].mean(axis=1).T
    plt.imsave("dec_1_1_attn_map.png", dec_1_1_attn_map)
    dec_1_2_attn_map = block_1_attn_maps[1][1][-1].mean(axis=1).T
    plt.imsave("dec_1_2_attn_map.png", dec_0_2_attn_map)

    print(decoder_maps[0].keys)