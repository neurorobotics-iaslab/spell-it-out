#%% PRELIMINARY IMPORTS
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from PIL import Image
from tqdm import tqdm
from wordcloud import WordCloud

from reproducibility.analysis.phonetic_statistics import calc_phone_freq
from spellitout import SpellItOutDataset
from spellitout.it_phonetics import (PhoneticCategory,
                                     load_phonitalia_phones,
                                     load_phonitalia_wordforms,
                                     string2phonlist)

dataset = SpellItOutDataset()
IMAGES_PATH = Path(__file__).resolve().parents[1] / 'results' / 'other_images'

bg_color = "#f8f7f2"

if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

#%% DRAW WORDCLOUD

df = dataset.wordlist

words = df['sentence'].tolist()
colors = ['#fe9829','#3498c3', '#67c2a5' ]
colors = ['#fe9829','#ff9062','#43d3ff','#3498c3',]
cmap = LinearSegmentedColormap.from_list('pipi', colors, N=6)
def plot_wordcloud(words,save_as=None,mask=None):
    # Join the words into a single string
    text = " ".join(words)


    # Create and configure the WordCloud
    wordcloud = WordCloud(
        # width=400, 
        # height=400,
        mask = mask,
        background_color=bg_color,
        colormap=cmap,
        collocations=False,
        random_state=2048
    ).generate(text)
    return wordcloud

mask = np.where(np.array(Image.open(f"{IMAGES_PATH}/cloud_mask.png"))>150,255,0)
words = df['sentence'].tolist()

phones = df['sentence_syll'].tolist()
phones = [p.replace('.',' ') for p in phones]
wordcloud = plot_wordcloud(phones,mask=mask)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig(IMAGES_PATH/'wordcloud_phones.png',dpi=300,bbox_inches='tight',pad_inches=0,transparent=True,facecolor=bg_color)
plt.savefig(IMAGES_PATH/'wordcloud_phones.svg',bbox_inches='tight',pad_inches=0,transparent=True,facecolor=bg_color)
plt.show()


#%% PLOT EXAMPLE EEG FROM TRUE DATA
raw = dataset.get_run(subject=1, task='overt',run=1).raw

raw.load_data()
raw.pick('eeg')
raw.load_data()
raw.filter(1,24)

plt.figure(figsize=(8,4))

n = 8
s = int(raw.info['sfreq']*4)
for i in range(n):
    plt.plot(i*100+1e6*raw.get_data()[i,:s],color="#003f77",linewidth=3)

#remove axes
plt.axis('off')
plt.tight_layout()
plt.savefig(IMAGES_PATH/'eeg_example.png',dpi=300,bbox_inches='tight',pad_inches=0,transparent=True)
plt.savefig(IMAGES_PATH/'eeg_example.svg',bbox_inches='tight',pad_inches=0,transparent=True)
plt.show()

#%% PLOT BINARY VAD EXAMPLE
plt.figure(figsize=(5,3))
n = 1000
t = np.arange(n)
v = np.zeros(n)
v[200:380] = 1
v[400:450] = 1
v[600:800] = 1
plt.plot(t,v,color="#003f77",linewidth=3)
# plt.text(-200,0,'Silence',fontsize=20,color='#00284b')
# plt.text(-200,0.9,'Speech',fontsize=20,color='#00284b')
plt.text(400,-0.2,'SILENCE',fontsize=20,color='#00284b',fontweight='bold')
plt.text(400,1.1,'SPEECH',fontsize=20,color='#00284b',fontweight='bold')
plt.hlines([-0.01,1.01],0,n,colors='#00284b',linestyles='dashed',linewidth=1)
plt.xlim(0,n)
plt.axis('off')
plt.tight_layout()
plt.savefig(IMAGES_PATH/'VAD_example.png',dpi=300,bbox_inches='tight',pad_inches=0,transparent=True)
plt.savefig(IMAGES_PATH/'VAD_example.svg',bbox_inches='tight',pad_inches=0,transparent=True)
plt.show()


#%% DRAW COMPARISON OF PHONEME FREQUENCIES

wordlist = dataset.wordlist

df = load_phonitalia_phones()

values = ['a', 'b', 'c', 'd', 'e', 'E', 'f', 'G', 'g', 'i', 'j', 'k', 'L',
       'l', 'm', 'n', 'N', 'O', 'o', 'p', 'r', 's', 'S', 't', 'u', 'v',
       'w', 'z', 'Z']

ipa_map = {
    'a': 'a', 'i': 'i', 'o': 'o', 'e': 'e', 'u': 'u', 'E': 'ɛ', 'O': 'ɔ',
    't': 't', 'r': 'r', 'n': 'n', 's': 's/z', 'l': 'l', 'k': 'k', 'm': 'm', 'p': 'p',
    'd': 'd', 'v': 'v', 'j': 'j', 'b': 'b', 'f': 'f', 'c': 'tʃ', 'z': 'ts', 'g': 'ʤ',
    'G': 'g', 'w': 'w', 'L': 'ʎ', 'Z': 'dz', 'S':'ʃ', 'N': 'ɲ'
    # add all mappings as needed from your table
}

phonitalia_counts = df['All TokenF'].values
phonitalia_counts = phonitalia_counts/df['All TokenF'].sum() * 100

v,c = calc_phone_freq(wordlist.sentence.values)
wordlist_counts =  np.array([c[v==val].item() if val in v else 0 for val in values])
wordlist_counts = wordlist_counts/wordlist_counts.sum() * 100

# Data assumed from your snippet
x = np.arange(len(values))
width = 0.55
# palette1 = sns.color_palette("BuGn_r",1)[0]
# palette2 = sns.color_palette("YlOrBr",1)[0]

# palette1 = "#9ac7e0"
# palette2 = "#4682b4"


palette2 = "#fe9829"
palette1 = "#009db3"

sorted_idx = [values.index(val) for val in ipa_map.keys()]
ipa_labels = [ipa_map.get(val, val) for val in values]

plt.figure(figsize=(15,5))
plt.gca().set_facecolor(bg_color)

# Remove all borders and grid
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.grid(False)

phonitalia_counts_sorted = values

# Bars
bars1 = plt.bar(x, phonitalia_counts[sorted_idx], width, color=palette1, label='Phonitalia', edgecolor='none', zorder=3)
bars2 = plt.bar(x, -wordlist_counts[sorted_idx], width, color=palette2, label='Our Wordlist', edgecolor='none', zorder=3)

# Add floating labels above and below bars
for idx, bar in enumerate(bars1):
    plt.text(bar.get_x()+width/2, bar.get_height()+0.8, list(ipa_map.values())[idx], ha='center', va='bottom', fontsize=12, color=palette1)
for idx, bar in enumerate(bars2):
    plt.text(bar.get_x()+width/2, bar.get_height()-1.0, list(ipa_map.values())[idx], ha='center', va='top', fontsize=12, color=palette2)

plt.yticks([])
plt.xticks([])
plt.xlabel('')
plt.ylabel('')
# plt.title('Phoneme Frequency Comparison', fontsize=18, weight='bold', color='#444443')
plt.legend(frameon=False, fontsize=13, loc='upper right')

plt.tight_layout()

plt.savefig(IMAGES_PATH/'phoneme_frequency_comparison.png', dpi=300, bbox_inches='tight', facecolor=bg_color)
plt.savefig(IMAGES_PATH/'phoneme_frequency_comparison.svg', bbox_inches='tight', facecolor=bg_color)
plt.show()

# VERTICAL VERSION


plt.figure(figsize=(5,9),facecolor=bg_color)
plt.gca().invert_yaxis()
plt.gca().set_facecolor(bg_color)
# Remove all borders and grid
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.grid(False)
# Bars
bars1 = plt.barh(x, -phonitalia_counts[sorted_idx], width, color=palette1, label='Phonitalia', edgecolor='none', zorder=3)
bars2 = plt.barh(x, wordlist_counts[sorted_idx], width, color=palette2, label='Our Wordlist', edgecolor='none', zorder=3)

# Add floating labels above and below bars
for idx, bar in enumerate(bars1):
    plt.text(bar.get_width()-1.0, bar.get_y()+width/2,  list(ipa_map.values())[idx], ha='right', va='center', fontsize=13, color=palette1)
for idx, bar in enumerate(bars2):
    plt.text(bar.get_width()+0.8, bar.get_y()+width/2, list(ipa_map.values())[idx], ha='left', va='center', fontsize=13, color=palette2)


plt.yticks([])
plt.xticks([])
plt.xlabel('')
plt.ylabel('')
# plt.title('Phoneme Frequency Comparison', fontsize=18, weight='bold', color='#444443')
plt.legend(frameon=False, fontsize=13, loc='upper center',ncols=2)

plt.tight_layout()

plt.savefig(IMAGES_PATH / 'phoneme_frequency_comparison_vertical.svg', bbox_inches='tight', facecolor=bg_color)
plt.savefig(IMAGES_PATH / 'phoneme_frequency_comparison_vertical.png', dpi=300, bbox_inches='tight', facecolor=bg_color)
plt.show()

#%% DRAW PHONEME CATEGORY MATRIX

# bars_color = 'steelblue'
# heatmap_color = 'Blues'


bars_color = "#289eb4"

color_palette = ['#67c2a5', '#3498c3', '#fe9829', '#e74c3c', '#9b59b6', '#f1c40f', '#e67e22', '#1abc9c', '#34495e']

heatmap_cmap = 'RdYlGn'

nature_seq_colors = [
    "#F4F8F8",  # Ice/Off-white
    "#A8DADC",  # Soft cyan/powder blue
    "#3A9AB2",  # Classic Nature teal
    "#1D6388",  # Mid-tone ocean blue
    "#0B3056",  # Scientific Data deep navy
    "#051329",  # Charcoal navy
]

heatmap_cmap = LinearSegmentedColormap.from_list(
    "NatureScientificData", nature_seq_colors, N=256
)

sound = dict()
sound[PhoneticCategory.NASAL]=0
sound[PhoneticCategory.OCCLUSIVE]=1
sound[PhoneticCategory.AFFRICATE]=2
sound[PhoneticCategory.FRICATIVE]=3
sound[PhoneticCategory.VIBRANT]=4
sound[PhoneticCategory.LATERAL]=5
sound[PhoneticCategory.APPROXIMANT]=6

sound[PhoneticCategory.HIGH]=7
sound[PhoneticCategory.MEDIUM_HIGH]=8
sound[PhoneticCategory.MEDIUM_LOW]=9
sound[PhoneticCategory.LOW]=10

position = dict()
position[PhoneticCategory.BILABIAL] = 0
position[PhoneticCategory.LABIODENTAL] = 1
position[PhoneticCategory.DENTIALVEOLAR] = 2
position[PhoneticCategory.POSTALVEOLAR] = 3
position[PhoneticCategory.PALATAL] = 4
position[PhoneticCategory.VELAR] = 5
position[PhoneticCategory.ANTERIOR] = 6
position[PhoneticCategory.CENTRAL] = 7
position[PhoneticCategory.POSTERIOR] = 8

cat_matrix = np.zeros((len(sound),len(position)))
valid_combinations = [(0,0),(0,2),(0,4),(1,0),(1,2),(1,5),(2,2),(2,3),(3,1),(3,2),(3,3),(4,2),(5,2),(5,4),(6,4),(6,5),(7,6),(7,8),(8,6),(8,8),(9,6),(9,8),(10,7)]

for _,row in wordlist.iterrows():
    phones = string2phonlist(row['sentence_syll'])
    for phone in phones:
        s,p = None,None
        cat = phone.categories        
        for c in cat: 
            if c in sound.keys(): s = sound[c]
            if c in position.keys(): p = position[c]
        if s is None or p is None:
            raise ValueError(f'Phone {phone} has no category, {cat},{sound.keys()},{position.keys()}')            
        cat_matrix[s,p] += 1


plt.figure(figsize=(12, 12))

# make a 2x2 grid
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.01, wspace=0.01)

# Horizontal bar plot
hbars = plt.subplot(gs[1, 1])
hbars.set_facecolor(bg_color)
hbars.barh(range(len(sound)), cat_matrix.sum(axis=1), color=bars_color)
hbars.set_yticks(range(len(sound)))
hbars.set_yticklabels([e.value for e in sound.keys()])
hbars.invert_yaxis()
hbars.axis('off')
for i, count in enumerate(cat_matrix.sum(axis=1)):
    hbars.text(count, i, int(count), ha='left', va='center', fontsize=13, fontweight='bold',)

# Vertical bar plot
vbars = plt.subplot(gs[0, 0])
vbars.set_facecolor(bg_color)
vbars.bar(range(len(position)), cat_matrix.sum(axis=0), color=bars_color)
vbars.set_xticks(range(len(position)))
vbars.set_xticklabels([e.value for e in position.keys()], rotation=90)
vbars.axis('off')
for i, count in enumerate(cat_matrix.sum(axis=0)):
    vbars.text(i, count, int(count), ha='center', va='bottom', fontsize=13, fontweight='bold')

# Heatmap
heatmap = plt.subplot(gs[1, 0], sharey=hbars, sharex=vbars)
heatmap.set_facecolor(bg_color)
im = heatmap.imshow(cat_matrix, cmap=heatmap_cmap, aspect='auto')
heatmap.set_xticks(range(len(position)))
heatmap.set_xticklabels([e.value for e in position.keys()], rotation=90, fontsize=13, fontweight='bold',)
heatmap.set_yticks(range(len(sound)))
heatmap.set_yticklabels([e.value for e in sound.keys()], fontsize=13, fontweight='bold',)

plt.vlines(5.5, -0.5, len(sound) - 0.5, color='black')
plt.hlines(6.5, -0.5, len(position) - 0.5, color='black')
# Add text annotations
for i in range(len(sound)):
    for j in range(len(position)):
        if (i, j) in valid_combinations:
            value = int(cat_matrix[i, j])
            color = 'white' if cat_matrix[i, j] > cat_matrix.max() / 2 else 'black'
            heatmap.text(j, i, value, ha='center', va='center', fontsize=13, fontweight='bold', color=color)
            
plt.savefig(IMAGES_PATH/'phoneme_category_matrix.png', dpi=300, bbox_inches='tight', facecolor=bg_color)
plt.savefig(IMAGES_PATH/'phoneme_category_matrix.svg', bbox_inches='tight', facecolor=bg_color)
plt.show()


#%% Print out the IPA representation of each sentence in the wordlist
wordforms = load_phonitalia_wordforms()
wordlist = dataset.wordlist
ipa_map = {
    'a': 'a', 
    'i': 'i', 
    'o': 'o', 
    'e': 'e', 
    'u': 'u', 
    'E': 'ɛ', 
    'O': 'ɔ',
    't': 't',
    'r': 'r', 
    'n': 'n', 
    's': 's/z', 
    'l': 'l', 
    'k': 'k', 
    'm': 'm', 
    'p': 'p',
    'd': 'd', 'v': 'v', 'j': 'j', 'b': 'b', 'f': 'f', 'c': 'tʃ', 'z': 'ts', 'g': 'ʤ',
    'G': 'g', 'w': 'w', 'L': 'ʎ', 'Z': 'dz', 
    
    'S':'ʃ', 
    'N': 'ɲ'
    # add all mappings as needed from your table
}

wordlist['syll_IPA'] = wordlist['sentence_syll'].apply(lambda x: ''.join(ipa_map.get(char, char) for char in x))

for index, row in wordlist.iterrows():
    n_words = len(row['sentence'].split())
    n_syllables = len(row['sentence_syll'].replace('.', ' ').split())
    print(f"Sentence: {row['sentence']}, IPA: {row['syll_IPA']}, Words: {n_words}, Syllables: {n_syllables}")


def is_buildable(phonesyll, syllables):
    """
    Check if a given phonesyll can be built from the syllables in the syllables list.
    """
    phonesyll_sylls = phonesyll.split('.')
    for syll in phonesyll_sylls:
        if syll not in syllables:
            return False
    return True

words = wordlist['sentence'].apply(lambda x: x.split(' ')).explode().unique()
syllables = wordlist['sentence_syll'].apply(lambda x: x.replace('.',' ').split(' ')).explode().unique()
composable_words = wordforms[wordforms['PhoneSyll'].apply(lambda x: is_buildable(x, syllables))].drop_duplicates(subset=['word'])
print(f"Number of unique words: {len(words)}")
print(f"Number of composable words: {len(composable_words)}")
print(f"Number of unique composable wordforms: {len(composable_words['word'])}")
