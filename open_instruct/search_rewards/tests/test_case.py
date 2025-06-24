test_case = [
    {
        "initial_prompt": "What publicly available datasets are typically used for evaluating type inference systems in python?",
        "metric_config": {
            "name": "rubric_corpusqa_generic",
            "config": {
                "question": "What publicly available datasets are typically used for evaluating type inference systems in python?",
                "low_length": 300,
                "high_length": 600,
                "length_weight": 0.05,
                "expertise_weight": 0.05,
                "citations_weight": 0.2,
                "excerpts_weight": 0.1,
                "other_properties": [
                    {
                        "name": "most_important_item_0",
                        "criterion": "Near the beginning, the answer should briefly define what is the goal of using a type inference system for programming languages in general.",
                        "weight": 0.13333333333333333,
                        "evidence": [
                            "Goal of type inference: Automatically deduce the most general type for each expression. Two key points: 1. Automatically inferring types: This means the programmer has to write no types, but still gets all the benefit from static typing 2. Inferring the most general type: This means we want to infer polymorphic types whenever possible"
                        ],
                    },
                    {
                        "name": "most_important_item_1",
                        "criterion": "The answer should emphasize on the importance of an automatic type inference system for Python.",
                        "weight": 0.13333333333333333,
                        "evidence": [
                            " its dynamic type system can lead to potential type errors, leading researchers to explore automatic type inference approaches for Python programs.",
                            "In Python, which is dynamically typed, this determination takes place at runtime. To address potential ambiguities, developers can utilize type annotations, which explicitly specifies the expected data types of variables or function returns. As the complexity of software projects increases, programmers find it increasingly challenging to maintain consistent data types. In response to this challenge, both industry and academia have developed type inference tools and static type checkers.",
                        ],
                    },
                    {
                        "name": "most_important_item_2",
                        "criterion": "The answer should discuss the need for a unified approach for evaluating different type inference systems and mention several evaluation metrics, including exact matches, report of missing types, accuracy, etc.",
                        "weight": 0.13333333333333333,
                        "evidence": [
                            "In light of the growing interest in type inference research for Python, both researchers and practitioners require a standardized process to assess the performance of various type inference techniques.",
                            "Exact matches: The number of inferred types that exactly match the ground truth. This metric is used widely used in the literature to evaluate type inference tools (Allamanis et al., 2020; Peng et al., 2022; Mir et al., 2022).",
                            "Report of missing types: List of types that are present in the ground truth but are unreported by the tools.",
                            "### **Accuracy** The correctness of the inferred types compared to the ground truth. ### **Performance** The computation resources and time required to run the type inference. ### **Coverage** The range and variety of code constructs and libraries handled by the inference system.",
                        ],
                    },
                    {
                        "name": "most_important_item_3",
                        "criterion": "The answer should enumerate publicly available datasets used for evaluating type inference systems in Python and provide a brief description for each of them.",
                        "weight": 0.13333333333333333,
                        "evidence": [
                            "1. **ManyTypes4Py**: - **Description**: ManyTypes4Py is a large Python dataset for machine learning-based type inference. It contains 5,382 Python projects with over 869,000 type annotations. The dataset is split into training, validation, and test sets by files to facilitate the training and evaluation of machine learning models. - **Features**: The dataset includes a lightweight static analyzer pipeline to extract type information from abstract syntax trees (ASTs) and store the results in JSON-formatted files.",
                            "The Typilus model [8] is accompanied by a dataset that contains 600 Python projects. Moreover, the source code files of Typilus' dataset are converted to graph representations that are only suitable for training the Typilus model.",
                            "2. **TypeEvalPy**: - **Description**: TypeEvalPy is a micro-benchmarking framework for evaluating type inference tools. It contains 154 code snippets with 845 type annotations across 18 categories targeting various Python features. - **Features**: The framework manages the execution of containerized tools, transforms inferred types into a standardized format, and produces meaningful metrics for assessment.",
                            "3. **BigQuery Public Datasets**: - **Description**: BigQuery provides a range of public datasets that can be used for various purposes, including type inference. These datasets are accessible through the Google Cloud Public Dataset Program and can be queried using SQL or GoogleSQL. - **Features**: The datasets include a variety of data sources, such as weather information, GitHub repository data, and Wikipedia revision history.",
                            "Raychev et al. [16] published the Python-150K dataset in 2016, which contains 8,422 Python projects.",
                            "Python-150K dataset [16] is not collected solely for the ML-based type inference task, meaning that a large number of projects in the dataset may not have type annotations at all, especially given the time that the dataset was created.",
                            "Our main dataset, BetterTypes4Py, is constructed by selecting a high-quality subset from the ManyTypes4Py dataset (Mir et al., 2021), which was used to train Type4Py.",
                            "InferTypes4Py, a test set derived from the source code of Typilus, Type4Py, and our own tool, none of which were used as CodeT5's (pre-)training data",
                        ],
                    },
                    {
                        "name": "nice_to_have_item_0",
                        "criterion": "The answer could explain different categories of methods for type inference in Python such as rule-based and ML-based approaches.",
                        "weight": 0.06666666666666667,
                        "evidence": [
                            "Existing type inference approaches can be generally grouped into three categories, i.e., rule-based, supervised, and cloze-style approaches. The rule-based type inference approaches can ensure the accuracy of predicted variable types, but they suffer from low coverage problems caused by dynamic features and external calls. Supervised type inference approaches, while feature-agnostic and able to mitigate the low coverage problem, require large, high quality annotated datasets and are limited to pre-defined types. As zero-shot approaches, the cloze-style approaches reformulate the type inference problem into a fill-in-the-blank problem by leveraging the general knowledge in powerful pre-trained code models. However, their performance is limited since they ignore the domain knowledge from static typing rules which reflect the inference logic."
                        ],
                    },
                ],
            },
        },
        "case_id": "d44280651a6fb71d56ee96834e180fa6",
        "annotator": "Annotator 1 Assignments",
        "agreement": True,
    },
]
