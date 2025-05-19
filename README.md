# language-classifier
This project is a 🌐 Wikipedia language classifier that determines whether a 15-word segment is in English or Dutch. 🇬🇧 🇳🇱

I compiled test data by scraping random Wikipedia articles and created features based on the data, such as the presence of language-specific function words and average word length. These features were used to train two learning algorithms: **decision tree learning** and **Adaboost** with decision trees as the base model.

To implement these algorithms, I did not rely on libraries for training. Instead, I implemented them from scratch, including the use of the [information gain algorithm](https://en.wikipedia.org/wiki/Information_gain_(decision_tree)) for decision-making in the tree construction.

This classifier has a 90% accuracy ✨
