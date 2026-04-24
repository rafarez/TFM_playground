# Context
Tabular Foundation Models (TFM) have emerged as a promising approach for learning
generalizable representations across diverse structured datasets. A critical component in
training these models is the design of the underlying architecture, which must efficiently capture the statistical properties and relational structures characteristic of real-world tables. However, the choice of architecture has profound implications for downstream performance. Architectures that are too simplistic may fail to capture the complex dependencies, mixed data types, and distributional irregularities found in practice, while overly complex architectures risk overfitting, poor generalization, or computational intractability. Understanding how different architectural design choices influence learned representations, transfer capabilities, and robustness to distribution shift is therefore essential for advancing tabular foundation models.

This project aims to improve the existing architecture of a TFM in order to enhance model
performance on new downstream tasks, without modifying the data generation pipeline used for pre-training.

# Objective

The project goal is to design and optimize the model architecture to improve performance on standard classification benchmarks, without modifying the synthetic data generation prior.

To work towards these objectives, you can base your work on either the TabPFN architecture [Hollmann et al., 2023; Hollmann et al., 2025] or the TabICL architecture [Qu et al., 2025] — or draw inspiration from both. As a practical starting point, you can use the TFM-playground repository from auto-ml, which implements a minimalistic version of TabPFN called nanoTabPFN that can be easily trained with minimal compute. For each benchmark you perform, we encourage you to use our model as a comparison through our API, as well as a standard nanoTabPFN trained model.

# Ressources:
- TFM Playground repo: https://github.com/automl/TFM-Playground/
- Seldon API (to compare our model to your model) :
  - Access Code (personal do not share): https://prediction.neuralk-ai.com/
  - Doc: https://docs.neuralk-ai.com/getting_started/introduction.html

# Submission guidelines
- Code: Provide clean, well-commented, and modular code. The project should be
self-contained and easy to run in a private github repo.
- Report: Provide a clear explanation of your methodology, design decisions, and
evaluation framework. Show a solid understanding of the challenges involved in
designing and optimizing efficient architectures for pre-training large tabular models.
- README: Include clear instructions on how to use your modified architecture, explaining
its components and design choices, and how to run any evaluation scripts.

# Evaluation criteria

We will evaluate your out-of-the-box idea for designing an architecture that is meaningful and effective for a tabular foundation model.

● Innovation and Creativity: Thoughtfulness in designing novel architectural components
or optimization strategies, and in devising a meaningful evaluation framework for the
architectural changes.
● Technical Execution: Correctness and robustness of the architecture implementation.
Overall code quality: clean, well-organized, and reusable.
● Analytical Thinking: A logical approach and well-reasoned explanations in the report.
The ability to reason about the complex relationship between architectural properties
and downstream utility for a tabular foundation model.