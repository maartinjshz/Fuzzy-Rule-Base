
## 📝 Introduction

This repository provides a framework for **Water Quality Assessment** using **Implicative Fuzzy Inference Systems**. Developed for the **IPMU 2026** conference, this project explores the application of formal fuzzy logic to environmental decision-making under uncertainty.

In this work, the [**Implicative model**](#deductive-interpretation) has been adopted instead of the more common [**Mamdani (conjunctive) model**](#mamdani1975), as the Implicative model yields mathematically superior results regarding **Monotonicity** and **Coherence**.

## Getting started

To get started, you should start by cloning the repositoy:
```
git clone https://github.com/your-username/Fuzzy-Rule-Base.git
cd Fuzzy-Rule-Base
```

Once the reposotory has been cloned, to make a new conda environment with python ```3.11```:
```
conda create -n <myenv> python=3.11 -y
conda activate <myenv>
```

Once the environment has been set adn activated, you can install the package by running:
```
pip install -e ".[test]"
pytest
```

## 📂 Project Structure

```text
.
├── src/                       # contains the project code
│   └── fuzzy_inference/       
│       ├── fuzzy_utils        # Functions for fuzzy logic
│       ├── payoff_function    # contains payoff functions
│       ├── rule_bases         # Contains Rule Bases systems 
│       └── utils              # contains utility functions (could be skipped)
│  
├── conference_material/       # Contains examples of usage              
│   ├── Data                   # Contains data of water quality
│   └── Notebooks              # Functions for fuzzy logic
│
├── tests/                    # Unit tests for logic and data integrity
├── pyproject.toml            # Project metadata and dependencies
└── README.md                 # Project documentation
```

## 💻 How to Use

The main feature of this repository is the straightforward implementation of the **Implicative Inference System**.

### 1. Import and Initialization
You can import the class and create a new system as follows:

```python
from fuzzy_inference.rule_bases.Implicative_model import ImplicativeInferenceSystem

# Define a new system
my_implicative_system = ImplicativeInferenceSystem()
```

### 2. Configuration Parameters

The `ImplicativeInferenceSystem` class accepts the following arguments for customization:

* **`rule_grid`**: A matrix defining the space for input function. Each point defines, where respective input function reachis its Core. (For now, the only option is ```Hat``` function, due to them satisfying ```Ruspini``` condition.)
* **`output_membership_degree`**: An array defining the grid where consequent membership functions reach their core (membership degree = 1). 
  * *Example:* `np.array([0, 0.25, 0.5, 0.75, 1])`
* **`number_of_grid_points`**: The number of grid points used for the discretization of each rule.
* **`type_of_output_funs`**: The membership function shape for the Consequent.
  * *Options:* `'Gaussian'`, `'Hat'`
* **`type_of_modifier`**: The type of modifiers applied to the fuzzy sets.
  * *Options:* `'atl'`, `'atm'`, `'altm'`
* **`type_of_tnorm`**: Defines the T-norm used for the implication logic.
  * *Options:* `'min'` (Gödel), `'luk'` (Łukasiewicz), `'prod'` (Product)

For now, it only has one mapping option from input membership function to the consequent, by taking a ```maximum```. Because this system assumes, that input membership functions have a partial order. For example, quality ```Good```, ```Medium``` and  ```Bad```. Thus it is possible to map it to the ```maximum``` or the worst/best quiality measure. It will be added later as an additional argument.


Because the arguemnts ```rule_grid``` specifies the intervals for each parameter, it could happen, that during an inference, a values that is outside that range is passed. In that case, the system simply maps back the value closes endpoint of the interval.

### 3. The Inference

To perform inference, simply:
```
my_implicative_systemper.form_inference(sample)
```
The sample should be a ```numpy.ndarray``` of values. The size should be the same as number of rows for ```rule_base```. For now, the inference returns the ```Center of Gravity``` defuziffication and the ```membership function``` itself. The ```Mean of Maxima``` will be added as additional argument, that is returned. For now, it is calculated from the ```membership function``` and the code can be seen used in ```conference_material/Notebooks/Using_implicative_base_IPMU2026.ipynb```. There seems to be some issues with it, becuase when defuzzifying payoff function values, the ```Mean of Maxima``` should be constant for different uncertanty values of ```Payoff``` and  ```Cleaning Cost```, but it has some small, but visible differences. 


## 🤝 Acknowledgments

> This research is funded by the **Latvian Council of Science**, project *"A fuzzy logic-based approach to the value of information estimation in optimal control problems under uncertainty with applications to ecological management"*, Project No. **lzp-2024/1-0188**.


## 📚 References

<a name="mamdani1975"></a>
* **[1]** Mamdani, E. H., & Assilian, S. (1975). **An experiment in linguistic synthesis with a fuzzy logic controller**. *International Journal of Man-Machine Studies*, 7(1), 1–13. [https://doi.org/10.1016/S0020-7373(75)80002-2](https://doi.org/10.1016/S0020-7373(75)80002-2)

<a name="implicative_model"></a>
* **[2]** Bodenhofer, U., Dankova, M., Stepnicka, M., & Novak, V. (2007). **A Plea for the Usefulness of the Deductive Interpretation of Fuzzy Rules in Engineering Applications**. *2007 IEEE International Fuzzy Systems Conference*, 1–6. [https://doi.org/10.1109/FUZZY.2007.4295600](https://doi.org/10.1109/FUZZY.2007.4295600)