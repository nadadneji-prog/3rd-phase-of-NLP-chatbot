"""
================================================================================
SKINCARE CHATBOT WITH NLP - DERMAi
================================================================================
PROJECT OVERVIEW:
This chatbot helps people with skincare questions using Natural Language
Processing (NLP) and Machine Learning. Created by Nada Neji, the project uses
a dataset of skincare to train and the dataset has 3 major columns: the query,
the category and the sub-category.

This version includes hyperparameter tuning using Grid Search to optimize
both Naive Bayes and Decision Tree classifiers.
"""

# ================================================================================
# IMPORTS
# ================================================================================

import sys


def check_and_import_packages():
    """Check if all required packages are installed and import them."""
    print("=" * 80)
    print("STEP 1: IMPORTING REQUIRED PACKAGES")
    print("=" * 80)
    print("Loading Python libraries needed for the chatbot...\n")

    try:
        print("→ Importing nltk...")
        import nltk
        print("  ✓ nltk imported")

        print("→ Importing pandas...")
        import pandas as pd
        print("  ✓ pandas imported")

        print("→ Importing numpy...")
        import numpy as np
        print("  ✓ numpy imported")

        print("→ Importing ssl...")
        import ssl
        print("  ✓ ssl imported")

        print("→ Importing pickle...")
        import pickle
        print("  ✓ pickle imported")

        # Import specific NLP tools from NLTK
        print("→ Importing NLTK components...")
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from nltk.stem import WordNetLemmatizer
        from nltk import pos_tag
        print("  ✓ NLTK components imported")

        # Import machine learning tools from scikit-learn
        print("→ Importing scikit-learn components...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        print("  ✓ scikit-learn components imported")

        print("\n✓ All packages imported successfully!\n")

        # Return all imports as a dictionary
        return {
            'nltk': nltk,
            'pd': pd,
            'np': np,
            'ssl': ssl,
            'pickle': pickle,
            'word_tokenize': word_tokenize,
            'stopwords': stopwords,
            'PorterStemmer': PorterStemmer,
            'WordNetLemmatizer': WordNetLemmatizer,
            'pos_tag': pos_tag,
            'TfidfVectorizer': TfidfVectorizer,
            'train_test_split': train_test_split,
            'GridSearchCV': GridSearchCV,
            'MultinomialNB': MultinomialNB,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'accuracy_score': accuracy_score,
            'classification_report': classification_report,
            'confusion_matrix': confusion_matrix
        }

    except ImportError as e:
        print(f"\n❌ ERROR: Missing required package!")
        print(f"   {e}\n")
        print("=" * 80)
        print("INSTALLATION INSTRUCTIONS:")
        print("=" * 80)
        print("\nOpen your terminal/command prompt and run ONE of these commands:\n")
        print("Option 1: pip install nltk pandas scikit-learn numpy")
        print("Option 2: pip3 install nltk pandas scikit-learn numpy")
        print("\nIf using PyCharm:")
        print("1. File > Settings > Project > Python Interpreter")
        print("2. Click the '+' button")
        print("3. Install: nltk, pandas, scikit-learn, numpy")
        print("=" * 80)
        sys.exit(1)


# ================================================================================
# SSL CONFIGURATION
# ================================================================================

def configure_ssl(ssl_module):
    """Configure SSL certificate settings for NLTK downloads."""
    print("=" * 80)
    print("STEP 2: FIXING SSL CERTIFICATE ISSUES")
    print("=" * 80)
    print("Configuring secure downloads for NLTK data...\n")

    try:
        _create_unverified_https_context = ssl_module._create_unverified_context
    except AttributeError:
        # This means SSL is already working fine
        pass
    else:
        # Apply the fix for SSL certificate verification
        ssl_module._create_default_https_context = _create_unverified_https_context

    print("✓ SSL configuration complete!\n")


# ================================================================================
# NLTK DATA DOWNLOAD
# ================================================================================

def download_nltk_data(nltk_module):
    """Download required NLTK datasets."""
    print("=" * 80)
    print("STEP 3: DOWNLOADING NLTK LANGUAGE DATA")
    print("=" * 80)
    print("Downloading language models (this happens only once)...\n")

    datasets = ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger',
                'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4']

    for dataset in datasets:
        print(f"→ Downloading {dataset}...")
        try:
            nltk_module.download(dataset, quiet=True)
            print(f"  ✓ {dataset} downloaded")
        except Exception as e:
            print(f"  ⚠ Warning: Could not download {dataset}: {e}")

    print("\n✓ NLTK data download complete!\n")


# ================================================================================
# DATA LOADING
# ================================================================================

def load_dataset(pd_module, filename='skincare_dataset.csv'):
    """Load the skincare dataset from CSV file."""
    print("=" * 80)
    print("STEP 4: LOADING THE SKINCARE DATASET")
    print("=" * 80)
    print("Reading skincare questions from CSV file...\n")

    try:
        # Read the CSV file into a pandas DataFrame
        df = pd_module.read_csv(filename)

        # Display information about the dataset
        print(f"✓ Dataset loaded successfully!")
        print(f"   → Total number of questions: {len(df)}")
        print(f"   → Original categories: {df['category'].nunique()}")
        print()

        return df

    except FileNotFoundError:
        print(f"❌ ERROR: '{filename}' not found!")
        print("   Make sure the CSV file is in the same folder as this script.\n")
        sys.exit(1)


# ================================================================================
# CATEGORY SIMPLIFICATION
# ================================================================================

def simplify_category(category):
    """Simplify category labels by grouping similar concerns."""
    category = str(category).lower()

    # Group all acne-related categories
    if 'acne' in category or 'pimple' in category:
        return 'acne'
    # Group dry skin categories
    elif 'dry' in category or 'flaky' in category:
        return 'dry_skin'
    # Group oily skin categories
    elif 'oily' in category:
        return 'oily_skin'
    # Combination skin (oily and dry areas)
    elif 'combination' in category:
        return 'combination_skin'
    # Sensitive or reactive skin
    elif 'sensitive' in category:
        return 'sensitive_skin'
    # Medical skin conditions
    elif 'eczema' in category or 'psoriasis' in category:
        return 'skin_conditions'
    # Dark spots and uneven skin tone
    elif 'hyperpigmentation' in category or 'dark' in category or 'spot' in category:
        return 'hyperpigmentation'
    # Anti-aging and wrinkles
    elif 'aging' in category or 'wrinkle' in category:
        return 'anti_aging'
    # Redness and rosacea
    elif 'rosacea' in category or 'redness' in category:
        return 'redness'
    # Product recommendations
    elif 'product' in category or 'ingredient' in category:
        return 'product_advice'
    # Skincare routine questions
    elif 'routine' in category:
        return 'routine_advice'
    # Everything else
    else:
        return 'general_skincare'


def process_categories(df):
    """Apply category simplification to the dataset."""
    print("=" * 80)
    print("STEP 5: SIMPLIFYING CATEGORIES")
    print("=" * 80)
    print("Grouping similar skin concerns together for better learning...\n")

    # Apply the simplification function to create a new column
    df['simplified_category'] = df['category'].apply(simplify_category)

    # Show the results
    print(f"✓ Categories simplified successfully!")
    print(f"   → Reduced from {df['category'].nunique()} to {df['simplified_category'].nunique()} categories")
    print()
    print("How many questions per category:")
    print(df['simplified_category'].value_counts())
    print()

    return df


# ================================================================================
# NLP FUNCTIONS
# ================================================================================

def preprocess_text(text, word_tokenize_func, stopwords_corpus, lemmatizer):
    """
    Preprocess text using NLP techniques.

    Steps:
    1. Tokenization
    2. Stopword removal
    3. Lemmatization
    """
    # Step 1: Convert to lowercase and tokenize
    tokens = word_tokenize_func(str(text).lower())

    # Step 2: Remove stopwords and keep only alphabetic words
    stop_words = set(stopwords_corpus.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # Step 3: Lemmatize each word
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Step 4: Join words back into a single string
    return ' '.join(tokens)


def show_nlp_techniques(text, word_tokenize_func, stopwords_corpus, pos_tag_func,
                        WordNetLemmatizer_class, PorterStemmer_class):
    """
    Demonstrate all 5 NLP techniques on user input.

    Techniques:
    1. Tokenization
    2. Stopword Removal
    3. POS Tagging
    4. Lemmatization
    5. Stemming
    """
    print(f"Original Query: '{text}'")
    print("-" * 80)

    # ──────────────────────────────────────────────────────────────────
    # TECHNIQUE 1: TOKENIZATION
    # ──────────────────────────────────────────────────────────────────
    tokens = word_tokenize_func(text)
    print(f"1. TOKENIZATION (splitting into words):")
    print(f"   {tokens}")

    # ──────────────────────────────────────────────────────────────────
    # TECHNIQUE 2: STOPWORD REMOVAL
    # ──────────────────────────────────────────────────────────────────
    stop_words = set(stopwords_corpus.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    print(f"\n2. STOPWORD REMOVAL (removing 'the', 'is', 'a', etc.):")
    print(f"   {filtered_tokens}")

    # ──────────────────────────────────────────────────────────────────
    # TECHNIQUE 3: POS TAGGING
    # ──────────────────────────────────────────────────────────────────
    pos_tags = pos_tag_func(tokens)
    print(f"\n3. POS TAGGING (identifying word types):")
    print(f"   {pos_tags[:6]}...")  # Show first 6 for clarity

    # ──────────────────────────────────────────────────────────────────
    # TECHNIQUE 4: LEMMATIZATION
    # ──────────────────────────────────────────────────────────────────
    lemmatizer = WordNetLemmatizer_class()
    lemmatized = [lemmatizer.lemmatize(word.lower()) for word in filtered_tokens]
    print(f"\n4. LEMMATIZATION (converting to base form):")
    print(f"   {lemmatized}")

    # ──────────────────────────────────────────────────────────────────
    # TECHNIQUE 5: STEMMING
    # ──────────────────────────────────────────────────────────────────
    stemmer = PorterStemmer_class()
    stemmed = [stemmer.stem(word.lower()) for word in filtered_tokens]
    print(f"\n5. STEMMING (cutting to word root):")
    print(f"   {stemmed}")
    print()

    return lemmatized


def setup_nlp_functions():
    """Initialize NLP functions and display setup confirmation."""
    print("=" * 80)
    print("STEP 6: SETTING UP NLP TECHNIQUES")
    print("=" * 80)
    print("Creating functions to process text using NLP...\n")
    print("✓ NLP functions created successfully!\n")


# ================================================================================
# DATA PREPROCESSING
# ================================================================================

def preprocess_dataset(df, word_tokenize_func, stopwords_corpus, WordNetLemmatizer_class):
    """Apply NLP preprocessing to all queries in the dataset."""
    print("=" * 80)
    print("STEP 7: PREPROCESSING ALL QUESTIONS IN DATASET")
    print("=" * 80)
    print(f"Applying NLP techniques to all {len(df)} questions...\n")

    # Create lemmatizer instance
    lemmatizer = WordNetLemmatizer_class()

    # Apply preprocessing to each query and store in new column
    df['processed_text'] = df['query'].apply(
        lambda x: preprocess_text(x, word_tokenize_func, stopwords_corpus, lemmatizer)
    )

    print("✓ All questions preprocessed!\n")

    # Show examples of before and after
    print("EXAMPLES OF TEXT PREPROCESSING:")
    print("-" * 80)
    for i in range(min(3, len(df))):
        print(f"\nExample {i + 1}:")
        print(f"   Original: {df['query'].iloc[i]}")
        print(f"   Processed: {df['processed_text'].iloc[i]}")
        print(f"   Category: {df['simplified_category'].iloc[i]}")
    print()

    return df


# ================================================================================
# INTERACTIVE DEMONSTRATION
# ================================================================================

def run_interactive_demo(word_tokenize_func, stopwords_corpus, pos_tag_func,
                         WordNetLemmatizer_class, PorterStemmer_class):
    """Run interactive NLP demonstration loop."""
    print("=" * 80)
    print("STEP 8: INTERACTIVE NLP DEMONSTRATION")
    print("=" * 80)
    print("\n💬 Your NLP is now ready!")
    print("\nTYPE any skincare question and see the 5 techniques in action")
    print("Type 'quit', 'exit', or 'bye' to end the demonstration.\n")
    print("=" * 80)
    print()

    # Main interactive loop
    while True:
        # Get user input
        user_question = input("Enter your question: ").strip()

        # Check if user wants to exit
        if user_question.lower() in ['quit', 'exit', 'bye', 'q', 'stop']:
            print("\n" + "=" * 80)
            print("Thank you for exploring NLP techniques!")
            print("You've learned: Tokenization, Stopword Removal, POS Tagging,")
            print("                Lemmatization, and Stemming")
            print("=" * 80)
            print("\n✅ NLP Demonstration Complete! 🎉\n")
            break

        # Skip empty input
        if not user_question:
            print("⚠️  Please enter some text.\n")
            continue

        # Show all NLP techniques on the user's question
        print("\n" + "=" * 80)
        show_nlp_techniques(
            user_question,
            word_tokenize_func,
            stopwords_corpus,
            pos_tag_func,
            WordNetLemmatizer_class,
            PorterStemmer_class
        )
        print("=" * 80)
        print()


# ================================================================================
# STEP 9: FEATURE EXTRACTION WITH TF-IDF
# ================================================================================

def create_tfidf_features(df, TfidfVectorizer_class):
    """
    Convert preprocessed text into numerical features using TF-IDF.

    TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numbers
    that machine learning algorithms can understand.
    """
    print("=" * 80)
    print("STEP 9: CREATING TF-IDF FEATURES")
    print("=" * 80)
    print("Converting text into numerical features for machine learning...\n")

    # Initialize the TF-IDF Vectorizer
    print("→ Initializing TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer_class(
        max_features=500,  # Use top 500 most important words
        min_df=2,  # Word must appear in at least 2 documents
        max_df=0.8  # Ignore words appearing in more than 80% of documents
    )

    # Fit and transform the preprocessed text
    print("→ Fitting vectorizer and transforming text...")
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['simplified_category']

    print(f"✓ TF-IDF features created successfully!")
    print(f"   → Feature matrix shape: {X.shape}")
    print(f"   → Number of samples: {X.shape[0]}")
    print(f"   → Number of features: {X.shape[1]}")
    print(f"   → Number of categories: {y.nunique()}")
    print()

    return X, y, vectorizer


# ================================================================================
# STEP 10: SPLIT DATA INTO TRAINING AND TESTING SETS
# ================================================================================

def split_data(X, y, train_test_split_func, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Training set: Used to teach the model
    Testing set: Used to evaluate how well the model learned
    """
    print("=" * 80)
    print("STEP 10: SPLITTING DATA INTO TRAIN AND TEST SETS")
    print("=" * 80)
    print(f"Dividing data: {int((1 - test_size) * 100)}% for training, {int(test_size * 100)}% for testing...\n")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split_func(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Ensure proportional representation of all categories
    )

    print(f"✓ Data split successfully!")
    print(f"   → Training samples: {X_train.shape[0]}")
    print(f"   → Testing samples: {X_test.shape[0]}")
    print()

    return X_train, X_test, y_train, y_test


# ================================================================================
# STEP 11: TRAIN NAIVE BAYES CLASSIFIER
# ================================================================================

def train_naive_bayes(X_train, y_train, MultinomialNB_class):
    """
    Train a Naive Bayes classifier.

    Naive Bayes is a probabilistic classifier based on Bayes' theorem.
    It's fast, simple, and works well for text classification.
    """
    print("=" * 80)
    print("STEP 11: TRAINING NAIVE BAYES CLASSIFIER")
    print("=" * 80)
    print("Training the Naive Bayes model on skincare questions...\n")

    # Initialize the Naive Bayes classifier
    print("→ Initializing Naive Bayes classifier...")
    nb_model = MultinomialNB_class()

    # Train the model
    print("→ Training model on training data...")
    nb_model.fit(X_train, y_train)

    print("✓ Naive Bayes model trained successfully!")
    print(f"   → Model type: Multinomial Naive Bayes")
    print(f"   → Number of classes learned: {len(nb_model.classes_)}")
    print(f"   → Categories: {list(nb_model.classes_)}")
    print()

    return nb_model


# ================================================================================
# STEP 12: TRAIN DECISION TREE CLASSIFIER
# ================================================================================

def train_decision_tree(X_train, y_train, DecisionTreeClassifier_class):
    """
    Train a Decision Tree classifier.

    Decision Tree creates a tree-like model of decisions.
    It's intuitive, interpretable, and doesn't require feature scaling.
    """
    print("=" * 80)
    print("STEP 12: TRAINING DECISION TREE CLASSIFIER")
    print("=" * 80)
    print("Training the Decision Tree model on skincare questions...\n")

    # Initialize the Decision Tree classifier
    print("→ Initializing Decision Tree classifier...")
    dt_model = DecisionTreeClassifier_class(
        max_depth=10,  # Limit tree depth to prevent overfitting
        min_samples_split=5,  # Minimum samples required to split a node
        min_samples_leaf=2,  # Minimum samples required at leaf node
        random_state=42
    )

    # Train the model
    print("→ Training model on training data...")
    dt_model.fit(X_train, y_train)

    print("✓ Decision Tree model trained successfully!")
    print(f"   → Model type: Decision Tree Classifier")
    print(f"   → Tree depth: {dt_model.get_depth()}")
    print(f"   → Number of leaves: {dt_model.get_n_leaves()}")
    print(f"   → Number of classes learned: {len(dt_model.classes_)}")
    print()

    return dt_model


# ================================================================================
# STEP 13: EVALUATE BOTH MODELS
# ================================================================================

def evaluate_models(nb_model, dt_model, X_test, y_test, accuracy_score_func,
                    classification_report_func, confusion_matrix_func):
    """
    Evaluate and compare both Naive Bayes and Decision Tree models.
    """
    print("=" * 80)
    print("STEP 13: EVALUATING MODEL PERFORMANCE")
    print("=" * 80)
    print("Testing both models on unseen data...\n")

    # ──────────────────────────────────────────────────────────────────
    # EVALUATE NAIVE BAYES
    # ──────────────────────────────────────────────────────────────────
    print("┌" + "─" * 78 + "┐")
    print("│ NAIVE BAYES CLASSIFIER RESULTS" + " " * 47 + "│")
    print("└" + "─" * 78 + "┘")

    # Make predictions
    y_pred_nb = nb_model.predict(X_test)

    # Calculate accuracy
    nb_accuracy = accuracy_score_func(y_test, y_pred_nb)
    print(f"\n📊 Accuracy: {nb_accuracy * 100:.2f}%")

    # Show detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report_func(y_test, y_pred_nb))

    # ──────────────────────────────────────────────────────────────────
    # EVALUATE DECISION TREE
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ DECISION TREE CLASSIFIER RESULTS" + " " * 45 + "│")
    print("└" + "─" * 78 + "┘")

    # Make predictions
    y_pred_dt = dt_model.predict(X_test)

    # Calculate accuracy
    dt_accuracy = accuracy_score_func(y_test, y_pred_dt)
    print(f"\n📊 Accuracy: {dt_accuracy * 100:.2f}%")

    # Show detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report_func(y_test, y_pred_dt))

    # ──────────────────────────────────────────────────────────────────
    # COMPARISON
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Naive Bayes Accuracy:    {nb_accuracy * 100:.2f}%")
    print(f"Decision Tree Accuracy:  {dt_accuracy * 100:.2f}%")

    if nb_accuracy > dt_accuracy:
        print(f"\n🏆 Winner: Naive Bayes (by {(nb_accuracy - dt_accuracy) * 100:.2f}%)")
        best_model = nb_model
        model_name = "Naive Bayes"
    elif dt_accuracy > nb_accuracy:
        print(f"\n🏆 Winner: Decision Tree (by {(dt_accuracy - nb_accuracy) * 100:.2f}%)")
        best_model = dt_model
        model_name = "Decision Tree"
    else:
        print(f"\n🤝 It's a tie! Both models have equal accuracy.")
        best_model = nb_model
        model_name = "Naive Bayes"

    print("=" * 80)
    print()

    return best_model, model_name, nb_accuracy, dt_accuracy


# ================================================================================
# STEP 14: HYPERPARAMETER TUNING - NAIVE BAYES
# ================================================================================

def tune_naive_bayes_grid_search(X_train, y_train, X_test, y_test,
                                 accuracy_score_func, GridSearchCV_class,
                                 MultinomialNB_class):
    """
    Perform Grid Search hyperparameter tuning for Naive Bayes.

    Grid Search tests different combinations of parameters to find the best ones.
    """
    print("=" * 80)
    print("STEP 14: HYPERPARAMETER TUNING - NAIVE BAYES (GRID SEARCH)")
    print("=" * 80)
    print("Searching for the best Naive Bayes parameters...\n")

    # Define parameter grid to search
    print("→ Defining parameter grid...")
    param_grid_nb = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],  # Smoothing parameter
        'fit_prior': [True, False]  # Whether to learn class prior probabilities
    }

    print(f"   Parameters to test: {param_grid_nb}")
    print(f"   Total combinations: {len(param_grid_nb['alpha']) * len(param_grid_nb['fit_prior'])}")

    # Initialize base model
    nb_base = MultinomialNB_class()

    # Perform Grid Search with cross-validation
    print("\n→ Running Grid Search with 5-fold cross-validation...")
    grid_search_nb = GridSearchCV_class(
        estimator=nb_base,
        param_grid=param_grid_nb,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,  # Use all available processors
        verbose=1
    )

    # Fit Grid Search
    grid_search_nb.fit(X_train, y_train)

    # Get best model
    best_nb_model = grid_search_nb.best_estimator_

    # Evaluate on test set
    y_pred_tuned = best_nb_model.predict(X_test)
    tuned_accuracy = accuracy_score_func(y_test, y_pred_tuned)

    # Display results
    print("\n✓ Grid Search completed!")
    print(f"\n📊 NAIVE BAYES RESULTS:")
    print(f"   Best Parameters: {grid_search_nb.best_params_}")
    print(f"   Best CV Score: {grid_search_nb.best_score_ * 100:.2f}%")
    print(f"   Test Accuracy (Tuned): {tuned_accuracy * 100:.2f}%")
    print()

    return best_nb_model, tuned_accuracy, grid_search_nb.best_params_


# ================================================================================
# STEP 15: HYPERPARAMETER TUNING - DECISION TREE
# ================================================================================

def tune_decision_tree_grid_search(X_train, y_train, X_test, y_test,
                                   accuracy_score_func, GridSearchCV_class,
                                   DecisionTreeClassifier_class):
    """
    Perform Grid Search hyperparameter tuning for Decision Tree.

    Grid Search tests different combinations of parameters to find the best ones.
    """
    print("=" * 80)
    print("STEP 15: HYPERPARAMETER TUNING - DECISION TREE (GRID SEARCH)")
    print("=" * 80)
    print("Searching for the best Decision Tree parameters...\n")

    # Define parameter grid to search
    print("→ Defining parameter grid...")
    param_grid_dt = {
        'max_depth': [5, 10, 15, 20, None],  # Maximum depth of tree
        'min_samples_split': [2, 5, 10],  # Minimum samples to split node
        'min_samples_leaf': [1, 2, 4],  # Minimum samples at leaf
        'criterion': ['gini', 'entropy']  # Split quality measure
    }

    print(f"   Parameters to test: {param_grid_dt}")
    total_combinations = (len(param_grid_dt['max_depth']) *
                          len(param_grid_dt['min_samples_split']) *
                          len(param_grid_dt['min_samples_leaf']) *
                          len(param_grid_dt['criterion']))
    print(f"   Total combinations: {total_combinations}")

    # Initialize base model
    dt_base = DecisionTreeClassifier_class(random_state=42)

    # Perform Grid Search with cross-validation
    print("\n→ Running Grid Search with 5-fold cross-validation...")
    print("   (This may take a few minutes...)")
    grid_search_dt = GridSearchCV_class(
        estimator=dt_base,
        param_grid=param_grid_dt,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,  # Use all available processors
        verbose=1
    )

    # Fit Grid Search
    grid_search_dt.fit(X_train, y_train)

    # Get best model
    best_dt_model = grid_search_dt.best_estimator_

    # Evaluate on test set
    y_pred_tuned = best_dt_model.predict(X_test)
    tuned_accuracy = accuracy_score_func(y_test, y_pred_tuned)

    # Display results
    print("\n✓ Grid Search completed!")
    print(f"\n📊 DECISION TREE RESULTS:")
    print(f"   Best Parameters: {grid_search_dt.best_params_}")
    print(f"   Best CV Score: {grid_search_dt.best_score_ * 100:.2f}%")
    print(f"   Test Accuracy (Tuned): {tuned_accuracy * 100:.2f}%")
    print()

    return best_dt_model, tuned_accuracy, grid_search_dt.best_params_


# ================================================================================
# STEP 16: COMPARE ORIGINAL VS TUNED MODELS
# ================================================================================

def compare_tuned_vs_original(nb_original_acc, dt_original_acc,
                              nb_tuned_acc, dt_tuned_acc,
                              nb_best_params, dt_best_params):
    """
    Compare the performance of original models vs hyperparameter-tuned models.
    """
    print("=" * 80)
    print("STEP 16: COMPARING ORIGINAL VS TUNED MODELS")
    print("=" * 80)
    print()

    # ──────────────────────────────────────────────────────────────────
    # NAIVE BAYES COMPARISON
    # ──────────────────────────────────────────────────────────────────
    print("┌" + "─" * 78 + "┐")
    print("│ NAIVE BAYES: ORIGINAL vs TUNED" + " " * 47 + "│")
    print("└" + "─" * 78 + "┘")

    nb_improvement = (nb_tuned_acc - nb_original_acc) * 100

    print(f"\nOriginal Model:")
    print(f"   → Accuracy: {nb_original_acc * 100:.2f}%")
    print(f"   → Parameters: Default (alpha=1.0, fit_prior=True)")

    print(f"\nTuned Model (Grid Search):")
    print(f"   → Accuracy: {nb_tuned_acc * 100:.2f}%")
    print(f"   → Best Parameters: {nb_best_params}")

    print(f"\n📈 Improvement: {nb_improvement:+.2f}%")
    if nb_improvement > 0:
        print(f"   ✓ Tuning improved performance!")
    elif nb_improvement < 0:
        print(f"   ⚠ Original was better (overfitting on tuned model)")
    else:
        print(f"   ≈ No change in performance")

    # ──────────────────────────────────────────────────────────────────
    # DECISION TREE COMPARISON
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ DECISION TREE: ORIGINAL vs TUNED" + " " * 45 + "│")
    print("└" + "─" * 78 + "┘")

    dt_improvement = (dt_tuned_acc - dt_original_acc) * 100

    print(f"\nOriginal Model:")
    print(f"   → Accuracy: {dt_original_acc * 100:.2f}%")
    print(f"   → Parameters: Default (max_depth=10, min_samples_split=5, etc.)")

    print(f"\nTuned Model (Grid Search):")
    print(f"   → Accuracy: {dt_tuned_acc * 100:.2f}%")
    print(f"   → Best Parameters: {dt_best_params}")

    print(f"\n📈 Improvement: {dt_improvement:+.2f}%")
    if dt_improvement > 0:
        print(f"   ✓ Tuning improved performance!")
    elif dt_improvement < 0:
        print(f"   ⚠ Original was better (overfitting on tuned model)")
    else:
        print(f"   ≈ No change in performance")

    # ──────────────────────────────────────────────────────────────────
    # FINAL COMPARISON
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 FINAL MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nORIGINAL MODELS:")
    print(f"   Naive Bayes:     {nb_original_acc * 100:.2f}%")
    print(f"   Decision Tree:   {dt_original_acc * 100:.2f}%")

    print(f"\nTUNED MODELS (with Grid Search):")
    print(f"   Naive Bayes:     {nb_tuned_acc * 100:.2f}%  ({nb_improvement:+.2f}%)")
    print(f"   Decision Tree:   {dt_tuned_acc * 100:.2f}%  ({dt_improvement:+.2f}%)")

    # Determine overall best model
    all_accuracies = {
        'NB Original': nb_original_acc,
        'NB Tuned': nb_tuned_acc,
        'DT Original': dt_original_acc,
        'DT Tuned': dt_tuned_acc
    }

    best_model_name = max(all_accuracies, key=all_accuracies.get)
    best_accuracy = all_accuracies[best_model_name]

    print(f"\n🏆 OVERALL WINNER: {best_model_name} with {best_accuracy * 100:.2f}%")
    print("=" * 80)
    print()

    # Return the best tuned model info
    if nb_tuned_acc >= dt_tuned_acc:
        return 'nb_tuned', nb_tuned_acc
    else:
        return 'dt_tuned', dt_tuned_acc


# ================================================================================
# STEP 17: INTERACTIVE CHATBOT
# ================================================================================

def run_chatbot(best_model, model_name, vectorizer, word_tokenize_func,
                stopwords_corpus, WordNetLemmatizer_class):
    """
    Run the interactive skincare chatbot using the best trained model.
    """
    print("=" * 80)
    print("STEP 17: INTERACTIVE SKINCARE CHATBOT")
    print("=" * 80)
    print(f"\n🤖 DERMAi Chatbot is ready! (Using {model_name} model)")
    print("\nASK me any skincare question and I'll categorize it!")
    print("Type 'quit', 'exit', or 'bye' to end the chat.\n")
    print("=" * 80)
    print()

    # Create lemmatizer for preprocessing
    lemmatizer = WordNetLemmatizer_class()

    # Main chatbot loop
    while True:
        # Get user input
        user_question = input("🧴 Your Question: ").strip()

        # Check if user wants to exit
        if user_question.lower() in ['quit', 'exit', 'bye', 'q', 'stop']:
            print("\n" + "=" * 80)
            print("Thank you for using DERMAi Skincare Chatbot!")
            print("Stay healthy and take care of your skin! 💙")
            print("=" * 80)
            print()
            break

        # Skip empty input
        if not user_question:
            print("⚠️  Please enter a question.\n")
            continue

        # Preprocess the user's question
        processed_question = preprocess_text(
            user_question,
            word_tokenize_func,
            stopwords_corpus,
            lemmatizer
        )

        # Transform to TF-IDF features
        question_features = vectorizer.transform([processed_question])

        # Make prediction
        predicted_category = best_model.predict(question_features)[0]

        # Get prediction probability
        prediction_proba = best_model.predict_proba(question_features)[0]
        confidence = max(prediction_proba) * 100

        # Display result
        print(f"\n🔍 Analysis:")
        print(f"   Category: {predicted_category}")
        print(f"   Confidence: {confidence:.2f}%")
        print(f"   Model: {model_name}")
        print()


# ================================================================================
# MAIN FUNCTION
# ================================================================================

def main():
    """Main function to orchestrate the entire NLP and ML pipeline with tuning."""

    print("\n🚀 Starting DERMAi Skincare Chatbot with Hyperparameter Tuning...\n")

    # Step 1: Import all required packages
    imports = check_and_import_packages()

    # Step 2: Configure SSL
    configure_ssl(imports['ssl'])

    # Step 3: Download NLTK data
    download_nltk_data(imports['nltk'])

    # Step 4: Load dataset
    df = load_dataset(imports['pd'])

    # Step 5: Simplify categories
    df = process_categories(df)

    # Step 6: Setup NLP functions
    setup_nlp_functions()

    # Step 7: Preprocess entire dataset
    df = preprocess_dataset(
        df,
        imports['word_tokenize'],
        imports['stopwords'],
        imports['WordNetLemmatizer']
    )

    # Step 8: Run interactive NLP demonstration
    run_interactive_demo(
        imports['word_tokenize'],
        imports['stopwords'],
        imports['pos_tag'],
        imports['WordNetLemmatizer'],
        imports['PorterStemmer']
    )

    # ══════════════════════════════════════════════════════════════════
    # MACHINE LEARNING PIPELINE STARTS HERE
    # ══════════════════════════════════════════════════════════════════

    # Step 9: Create TF-IDF features
    X, y, vectorizer = create_tfidf_features(df, imports['TfidfVectorizer'])

    # Step 10: Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        imports['train_test_split']
    )

    # Step 11: Train Naive Bayes classifier (ORIGINAL)
    nb_model = train_naive_bayes(
        X_train, y_train,
        imports['MultinomialNB']
    )

    # Step 12: Train Decision Tree classifier (ORIGINAL)
    dt_model = train_decision_tree(
        X_train, y_train,
        imports['DecisionTreeClassifier']
    )

    # Step 13: Evaluate both ORIGINAL models
    best_model, model_name, nb_original_acc, dt_original_acc = evaluate_models(
        nb_model, dt_model,
        X_test, y_test,
        imports['accuracy_score'],
        imports['classification_report'],
        imports['confusion_matrix']
    )

    # ══════════════════════════════════════════════════════════════════
    # HYPERPARAMETER TUNING WITH GRID SEARCH
    # ══════════════════════════════════════════════════════════════════

    # Step 14: Tune Naive Bayes with Grid Search
    nb_tuned_model, nb_tuned_acc, nb_best_params = tune_naive_bayes_grid_search(
        X_train, y_train, X_test, y_test,
        imports['accuracy_score'],
        imports['GridSearchCV'],
        imports['MultinomialNB']
    )

    # Step 15: Tune Decision Tree with Grid Search
    dt_tuned_model, dt_tuned_acc, dt_best_params = tune_decision_tree_grid_search(
        X_train, y_train, X_test, y_test,
        imports['accuracy_score'],
        imports['GridSearchCV'],
        imports['DecisionTreeClassifier']
    )

    # Step 16: Compare original vs tuned models
    best_model_type, best_tuned_acc = compare_tuned_vs_original(
        nb_original_acc, dt_original_acc,
        nb_tuned_acc, dt_tuned_acc,
        nb_best_params, dt_best_params
    )

    # Select the best tuned model for chatbot
    if best_model_type == 'nb_tuned':
        final_model = nb_tuned_model
        final_model_name = "Naive Bayes (Tuned)"
    else:
        final_model = dt_tuned_model
        final_model_name = "Decision Tree (Tuned)"

    # Step 17: Run interactive chatbot with BEST TUNED model
    run_chatbot(
        final_model, final_model_name, vectorizer,
        imports['word_tokenize'],
        imports['stopwords'],
        imports['WordNetLemmatizer']
    )


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    main()