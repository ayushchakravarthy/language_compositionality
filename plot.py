import json
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    attn_file = 'results/language_parser/train_defaults_jump_attn_maps0.pickle'
    res_file = 'results/language_parser/train_defaults_jump0.json'

    with open(attn_file, 'rb') as f:
        attn_maps = pickle.load(f)

    with open(res_file, 'r') as f:
        stats = json.load(f)

    plt.plot(stats.get('test_accs'))
    plt.show()
    print(attn_maps)


