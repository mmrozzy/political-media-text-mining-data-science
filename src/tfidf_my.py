import pandas as pd              
import numpy as np               
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  
import re                       
from collections import defaultdict 
import matplotlib.pyplot as plt  
import seaborn as sns            
import os                        

# NER import - optional
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    NER_AVAILABLE = True
except (ImportError, OSError):
    NER_AVAILABLE = False

def normalize_text_ner(text):
    if not NER_AVAILABLE:
        return str(text).lower()  # fallback to basic normalization
    
    if pd.isna(text):
        raise ValueError("Error with normalization text.")
    
    text = str(text)
    doc = nlp(text)
    
    # normalise
    for ent in reversed(doc.ents): 
        if ent.label_ == "PERSON":
            if "trump" in ent.text.lower():
                text = text[:ent.start_char] + "trump" + text[ent.end_char:]
            elif "newsom" in ent.text.lower():
                text = text[:ent.start_char] + "newsom" + text[ent.end_char:]
            elif "biden" in ent.text.lower():
                text = text[:ent.start_char] + "biden" + text[ent.end_char:]
            else:
                # to lowercase for others
                text = text[:ent.start_char] + ent.text.lower() + text[ent.end_char:]
        elif ent.label_ == "GPE":  # geopolitical entities (entities, countries, etc.)
            if "francisco" in ent.text.lower():
                text = text[:ent.start_char] + "san francisco" + text[ent.end_char:]
            elif "california" in ent.text.lower():
                text = text[:ent.start_char] + "california" + text[ent.end_char:]
            else:
                text = text[:ent.start_char] + ent.text.lower() + text[ent.end_char:]
        elif ent.label_ == "ORG":  # organizations
            if "democratic" in ent.text.lower() and "party" in ent.text.lower():
                text = text[:ent.start_char] + "democratic party" + text[ent.end_char:]
            elif "republican" in ent.text.lower() and "party" in ent.text.lower():
                text = text[:ent.start_char] + "republican party" + text[ent.end_char:]
            else:
                text = text[:ent.start_char] + ent.text.lower() + text[ent.end_char:]
    
    return text.lower()

def normalize_text(text):
    if pd.isna(text):
        raise ValueError("Error with normalization text.")

    text = str(text).lower()

    # developed based on pre entity mapping tf-idf results
    entity_mappings = {
        # Political figures
        'trump_entity': ['trump', 'donald trump', 'president trump', 'former president trump', 'president donald trump', 'donald'],
        'newsom_entity': ['newsom', 'gavin newsom', 'governor newsom', 'gov newsom', 'california governor', 'gov gavin', 'gavin', 'gov', 'governor'],
        'biden_entity': ['biden', 'joe biden', 'president biden', 'biden administration', 'joe'],
        'harris_entity': ['harris', 'kamala harris', 'vice president harris', 'vp harris', 'kamala'],
        'williamson_entity': ['chief of staff', 'dana williamson', 'williamson', 'chief staff', 'chief', 'staff'],

        # Locations
        'california_entity': ['california', 'calif', 'ca', 'golden state', 'west coast'],
        'san_francisco_entity': ['san francisco', 'sf', 'san fran', 'francisco', 'san'],
        'texas_entity': ['texas', 'tx', 'lone star state'],
        'washington_entity': ['washington', 'dc', 'washington dc', 'capitol'],
        
        # Political terms
        'democrat_entity': ['democrat', 'democratic', 'democrats', 'dem', 'dems'],
        'republican_entity': ['republican', 'republicans', 'gop', 'rep', 'reps'],
        'government_entity': ['government', 'federal government', 'administration', 'govt'],
        'congress_entity': ['congress', 'congressional', 'house', 'senate', 'legislature'],
        
        # Issues
        'climate_entity': ['climate', 'climate change', 'global warming', 'environmental'],
        'economy_entity': ['economy', 'economic', 'economics', 'financial', 'fiscal'],
        'immigration_entity': ['immigration', 'immigrant', 'immigrants', 'undocumented', 'border'],
        'healthcare_entity': ['healthcare', 'health care', 'medical', 'hospital', 'insurance'],
        'planned_parenthood_entity': ['planned parenthood', 'parenthood']
    }

    for entity_name, variations in entity_mappings.items():
        for variation in variations:
            pattern = r'\b' + re.escape(variation) + r'\b'
            text = re.sub(pattern, entity_name, text) # replacement

    return text


def clean_text(text, use_ner=False):
    if pd.isna(text): return ""
    
    if use_ner:
        text = normalize_text_ner(text)
    else:
        text = normalize_text(text)
    
    text = re.sub(r'[^a-zA-Z\s_]', ' ', text)
    text = ' '.join(text.split())
    return text

number_idf_words = 10
def get_top_tfidf_word(texts, n_words = number_idf_words):
    custom_stop_words = set(ENGLISH_STOP_WORDS).union({
        # common reporting words
        'said', 'says', 'would', 'could', 'also', 'one', 'two', 'new', 'year', 
        'years', 'time', 'first', 'last', 'way', 'people', 'state', 'states',
        'according', 'report', 'news', 'like', 'make', 'made', 'get', 'go',
        # time-related words that appear frequently but aren't topically important
        'week', 'day', 'days', 'today', 'yesterday', 'monday', 'tuesday', 'wednesday',
        'thursday', 'friday', 'saturday', 'sunday', 'january', 'february', 'march',
        'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'
    })

    vectorizer = TfidfVectorizer(
        max_features=1000,              # top 1000 features - reduce noise
        stop_words=list(custom_stop_words),
        min_df=2,                       # word must appear in at least 2 documents - reduce noise
        max_df=0.8,                     # word must not appear in more than 80% of documents - removes very common words
        # no need for ngram since we normalise first    
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    word_scores = list(zip(feature_names, mean_scores)) # pair up
    word_scores.sort(key=lambda x: x[1], reverse=True)

    cleaned_scores = []
    for word, score in word_scores[:n_words*2]:
        # "trump_entity" becomes "trump"
        display_word = word.replace('_entity', '').replace('_', ' ')
        
        existing_words = [w[0] for w in cleaned_scores]
        if display_word not in existing_words:
            cleaned_scores.append((display_word, score))
        
        if len(cleaned_scores) >= n_words:
            break
    
    return cleaned_scores


def analyze_categories(df, use_ner=False):
    results = {}
    categories = df['Categories'].unique()

    for category in categories:
        if pd.isna(category): continue
        category_data = df[df['Categories'] == category] #filter
        
        texts = []
        for _, row in category_data.iterrows():
            title = clean_text(row['title'], use_ner=use_ner)
            body = clean_text(row['body'], use_ner=use_ner)
            combined_text = f"{title} {body}"
            if combined_text.strip():
                texts.append(combined_text)

        if len(texts) < 2:
            print(f"Category '{category}': Not enough documents ({len(texts)}) for meaningful TF-IDF analysis")
            continue

        top_words = get_top_tfidf_word(texts)
        results[category] = top_words

    return results

def group_political_leaning(leaning):
    if pd.isna(leaning):
        return 'Neutral'
    
    leaning = str(leaning).lower()
    
    if 'left' in leaning:
        return 'Left'
    elif 'right' in leaning:
        return 'Right'
    elif 'centrist' in leaning:
        return 'Neutral'
    else:
        return 'Neutral'

def analyze_categories_by_political_leaning(df, use_ner=False):
    results = {}
    
    if 'publisher_leaning' not in df.columns:
        print("No 'publisher_leaning' column found. Skipping political leaning analysis.")
        return results
    
    df['grouped_leaning'] = df['publisher_leaning'].apply(group_political_leaning)
    
    print("\nAnalyzing TF-IDF by political leaning...")
    print("=" * 60)
    
    leaning_counts = df['grouped_leaning'].value_counts()
    print(f"Article distribution by political leaning:")
    for leaning, count in leaning_counts.items():
        print(f"  {leaning}: {count} articles")
    print()
    
    leanings = df['grouped_leaning'].unique()
    for leaning in ['Left', 'Right', 'Neutral']: 
        if leaning not in leanings:
            continue
            
        print(f"\nProcessing {leaning} leaning...")
        
        leaning_data = df[df['grouped_leaning'] == leaning]
        
        texts = []
        for _, row in leaning_data.iterrows():
            title = clean_text(row['title'], use_ner=use_ner)
            body = clean_text(row['body'], use_ner=use_ner)
            combined_text = f"{title} {body}"
            if combined_text.strip():
                texts.append(combined_text)
        
        if len(texts) < 2:
            print(f"  Not enough documents ({len(texts)}) for analysis")
            continue
        
        top_words = get_top_tfidf_word(texts, n_words=number_idf_words)
        results[leaning] = top_words
        
        print(f"  Processed {len(texts)} articles")
        print(f"  Top 3 words: {', '.join([word for word, score in top_words[:3]])}")
    
    return results

def save_leaning_results_to_file(results, use_ner=False):

    actual_ner_used = use_ner and NER_AVAILABLE
    method_suffix = "ner" if actual_ner_used else "manual"
    
    import os
    viz_folder = "Visualizations"
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    
    filename = os.path.join(viz_folder, f'tfidf_political_leaning_{method_suffix}.txt')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("TF-IDF ANALYSIS BY POLITICAL LEANING\n")
        f.write("=" * 60 + "\n")
        f.write(f"Normalization method: {'NER' if actual_ner_used else 'Manual'}\n")
        f.write("=" * 60 + "\n\n")
        
        for leaning, top_words in results.items():
            f.write(f"Political Leaning: {leaning}\n")
            f.write("-" * 40 + "\n")
            
            for i, (word, score) in enumerate(top_words, 1):
                f.write(f"{i:2d}. {word:<25} {score:.4f}\n")
            
            f.write("\n" + "=" * 40 + "\n")
    
    print(f"\nPolitical leaning analysis saved as '{filename}'")

def create_leaning_visualization(results, use_ner=False):
    if not results:
        print("No political leaning results to visualize")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    leaning_colors = {
        'Left': '#1f77b4',      # Blue
        'Right': '#d62728',     # Red  
        'Neutral': '#7f7f7f'    # Gray
    }
    
    leaning_order = ['Left', 'Right', 'Neutral']
    
    for i, leaning in enumerate(leaning_order):
        if leaning not in results or not results[leaning]:
            axes[i].text(0.5, 0.5, f'No {leaning} data', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{leaning} Leaning', fontweight='bold')
            continue
        
        words_scores = results[leaning]
        words, scores = zip(*words_scores)
        
        y_pos = range(len(words))
        bars = axes[i].barh(y_pos, scores, color=leaning_colors[leaning], alpha=0.8)
        
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(words)
        axes[i].set_xlabel('TF-IDF Score')
        axes[i].set_title(f'{leaning} Leaning', fontweight='bold', fontsize=12)
        axes[i].invert_yaxis()
        axes[i].grid(True, alpha=0.3)
        
        for j, (bar, score) in enumerate(zip(bars, scores)):
            axes[i].text(score + max(scores) * 0.01, j, f'{score:.3f}', 
                        va='center', fontsize=9)
    
    actual_ner_used = use_ner and NER_AVAILABLE
    method_suffix = "ner" if actual_ner_used else "manual"
    
    viz_folder = "Visualizations"
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    
    filename = os.path.join(viz_folder, f'tfidf_political_leaning_{method_suffix}.png')
    
    plt.suptitle('TF-IDF Analysis by Political Leaning', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Political leaning visualization saved as '{filename}'")



def create_visualizations(results, use_ner=False):
    if not results:
        print("No results to visualize")
        return
    
    num_categories = len(results)
    if num_categories <= 3:
        rows, cols = 1, num_categories
        figsize = (5 * cols, 6)
    elif num_categories <= 6:
        rows, cols = 2, 3
        figsize = (15, 10)
    else:
        rows, cols = 3, 3
        figsize = (15, 15)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if num_categories == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    for i, (category, top_words) in enumerate(results.items()):
        if i >= len(axes):
            break
            
        words_scores = top_words
        if not words_scores:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(category)
            continue
            
        words, scores = zip(*words_scores)
        
        y_pos = range(len(words))
        bars = axes[i].barh(y_pos, scores, color=plt.cm.Set3(i % 12))
        
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(words)
        axes[i].set_xlabel('TF-IDF Score')
        axes[i].set_title(f'{category}', fontweight='bold', fontsize=11)
        axes[i].invert_yaxis() 

        for j, (bar, score) in enumerate(zip(bars, scores)):
            axes[i].text(score + max(scores) * 0.01, j, f'{score:.3f}', 
                        va='center', fontsize=9)
    
    for i in range(len(results), len(axes)):
        axes[i].remove()
    
   
    actual_ner_used = use_ner and NER_AVAILABLE
    method_suffix = "ner" if actual_ner_used else "manual"
    
    import os
    viz_folder = "Visualizations"
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    
    filename = os.path.join(viz_folder, f'tfidf_visualization_{method_suffix}.png')
    
    plt.suptitle('Top 10 TF-IDF Terms per Coverage Category', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()
    print(f"\nVisualization saved as '{filename}'")

def main():
    USE_NER = False  # True to use Named Entity Recognition, False for manual normalization
    
    print(f"Using {'NER (Named Entity Recognition)' if USE_NER else 'Manual Entity Normalization'}")
    if USE_NER and not NER_AVAILABLE:
        print("NER requested but not available. Falling back to manual normalization.")
        USE_NER = False
        
    try:
        import os
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'data_annotated_with_leaning.csv')
        
        if os.path.exists('data_annotated_with_leaning.csv'):
            df = pd.read_csv('data_annotated_with_leaning.csv')
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError("data_annotated_with_leaning.csv not found")
        
        print(f"Loaded {len(df)} articles")
    except FileNotFoundError:
        print("Error: data_annotated_with_leaning.csv not found in current directory or script directory")
        print("Please run the add_publisher_leaning.py script first to generate this file.")
        return  
    except Exception as e:
        print(f"Error loading data: {e}")
        return  
    
    print(f"Dataset shape: {df.shape}")                   
    print(f"Categories: {df['Categories'].nunique()} unique categories")  
    print(f"Articles per category:")
    print(df['Categories'].value_counts())                 
    print()  
    
    # run analyses
    print("\n" + "=" * 80)
    print("STARTING TF-IDF ANALYSIS")
    print("=" * 80)
    
    # regular category analysis
    results = analyze_categories(df, use_ner=USE_NER)
    
    print("\n" + "=" * 60)
    print("TF-IDF RESULTS BY CATEGORY")
    print("=" * 60)
    
    for category, top_words in results.items():
        print(f"\nCategory: {category}")
        print("-" * 40)
        print("Top words by TF-IDF score:")
        
        for i, (word, score) in enumerate(top_words, 1):
            print(f"{i:2d}. {word:<25} {score:.4f}")
    
    print("\n" + "=" * 60)
    
    # political leaning analysis
    leaning_results = analyze_categories_by_political_leaning(df, use_ner=USE_NER)
    
    # print leaning results
    if leaning_results:
        print("\n" + "=" * 60)
        print("TF-IDF RESULTS BY POLITICAL LEANING")
        print("=" * 60)
        
        for leaning, top_words in leaning_results.items():
            print(f"\nPolitical Leaning: {leaning}")
            print("-" * 40)
            print("Top words by TF-IDF score:")
            
            for i, (word, score) in enumerate(top_words, 1):
                print(f"{i:2d}. {word:<25} {score:.4f}")
        
        print("\n" + "=" * 60)
    
    # visualization for regular analysis
    try:
        create_visualizations(results, use_ner=USE_NER)
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # save political leaning analysis to file
    try:
        save_leaning_results_to_file(leaning_results, use_ner=USE_NER)
    except Exception as e:
        print(f"Error saving political leaning analysis: {e}")
    
    # create political leaning visualization
    try:
        create_leaning_visualization(leaning_results, use_ner=USE_NER)
    except Exception as e:
        print(f"Error creating political leaning visualization: {e}")

if __name__ == "__main__":
    main()