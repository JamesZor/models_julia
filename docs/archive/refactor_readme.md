# BayesianFootball.jl - High-Level Design & Refactoring Plan

1. Core Philosophy

The BayesianFootball.jl package is being refactored from a monolithic structure into a modular architecture based on the principle of separation of concerns.
The goal is to create a collection of independent, interoperable sub-modules that communicate through clear, stable Application Programming Interfaces (APIs).

This design will make the system easier to maintain, test, and extend.
It enables a "plug-in" architecture where developers can easily add new models, data sources, or evaluation metrics without causing cascading changes throughout the codebase.

2. GitHub Repository Strategy: Monorepo

The project will be managed as a monorepo (a single GitHub repository) for the following reasons:

    Simplicity: All code lives in one place, simplifying dependency management.

    Atomic Commits: A single commit can span multiple modules, making it easy to implement cross-cutting features.

    Easier Collaboration: New contributors can get the entire project up and running with a single clone.

3. Proposed Module Architecture

The package will be organized into the following sub-modules, each corresponding to a directory within the src/ folder.

``` src/ |-- BayesianFootball.jl # Main module to bring everything together
         | |-- Data/ | |-- Data.jl # Sub-module for all data loading and handling 
            | |-- loader.jl # Functions to load data from CSVs, APIs, etc. 
         | -- types.jl # Structs for raw data (e.g. RawMatch) |
       |-- Features/ | |-- Features.jl # Sub-module for feature engineering | -- core.jl # Core feature creation logic | |-- Models/ | |-- Models.jl # Sub-module defining model interfaces and implementations | |-- interfaces.jl # Abstract types (e.g., AbstractPregameModel) | |-- pregame/ # Directory for all pre-game model definitions | | -- ar1_neg_bin.jl | -- ingame/ # Directory for all in-game model definitions | |-- Training/ | |-- Training.jl # Sub-module for model training pipelines | |-- api.jl # High-level API (e.g., train(model, data)) | -- backends/ # Specific training logic (e.g., for Turing.jl) | -- turing.jl | |-- Prediction/ | |-- Prediction.jl # Sub-module for generating predictions | |-- api.jl # High-level API (e.g., predict(model, features)) | -- types.jl # Prediction-specific structs (e.g. MatchPredictions) | |-- Evaluation/ | |-- Evaluation.jl # Sub-module for model evaluation | |-- metrics.jl # Scoring rules (RPS, ECE), etc. | -- betting.jl # Kelly criterion, staking strategies, etc. | -- Utils/ -- Utils.jl # General utility functions used across modules ```

4. Module Responsibilities & APIs

Data Module

    Responsibility: All I/O operations. This module is the gateway for getting data into the system.

    Core API:

        load_matches(source::String)::DataFrame

        load_odds(source::String)::DataFrame

Features Module

    Responsibility: Transforming raw data into model-ready features.

    Core API:

        create_features(model::AbstractFootballModel, df::DataFrame)::NamedTuple

Models Module

    Responsibility: Defining the model structures and interfaces. This is the core of the "plug-in" architecture.

    Core API (Interfaces):

        abstract type AbstractFootballModel end

        abstract type AbstractPregameModel <: AbstractFootballModel end

        abstract type AbstractInGameModel <: AbstractFootballModel end

        required_features(model::AbstractFootballModel)::Vector{Symbol}

        build_turing_model(model::AbstractFootballModel, features, outcomes)

Training Module

    Responsibility: Handling the model training process (e.g., running the MCMC sampler).

    Core API:

        train(model::AbstractFootballModel, features::NamedTuple, data::DataFrame)::Chains

Prediction Module

    Responsibility: Generating predictions from a trained model.

    Core API:

        predict(model::AbstractFootballModel, chains::Chains, features::NamedTuple)::MatchPredictions

Evaluation Module

    Responsibility: Scoring predictions and evaluating betting strategies.

    Core API:

        calculate_rps(predictions::MatchPredictions, outcomes::DataFrame)::Float64

5. Module Interactions & API Design Patterns

The modules interact through Julia's multiple dispatch. Instead of large if/else blocks, we define multiple small methods for the same function name, each specialized on a different model type. This makes the system extensible and easy to reason about.

Features ↔️ Models

Each model declares the features it needs by implementing a required_features(::MyModel). The Features module uses this information to call the correct feature-generation logic, dispatched on the model type.

```julia

Model declares its needs

Models.required_features(::AR1NegativeBinomial) = [:global_round, :team_ids]

Feature module fulfills the request

function create_features(model::AbstractFootballModel, data)
needed = Models.required_features(model)
return _generate(model, needed, data) # Dispatches to a specialized _generate method
end
```

Training & Prediction ↔️ Models

The Training and Prediction modules provide the high-level API, but delegate the model-specific implementation back to the model itself.

    Each model file implements its own Models.build_turing_model(::MyModel, ...) which contains the @model block.

    Each model file implements its own Prediction.predict(::MyModel, ...) which contains the optimized prediction logic.

The Training.train and Prediction.predict functions are simple, generic wrappers that call these specialized methods, which Julia's compiler selects automatically at runtime.

6. Type Management Strategy

Type definitions will be distributed into the modules that are their logical owners.

    Data/types.jl: Contains structs for raw data (e.g., RawMatch).

    Models/interfaces.jl: Contains the abstract types that define the model contracts (e.g., AbstractPregameModel).

    Prediction/types.jl: Contains the structs that represent the output of a prediction (e.g., MatchPredictions, MatchFTPredictions).

This avoids a single, monolithic types.jl file and ensures each module is a self-contained unit.

7. Example Workflow

A typical pre-game modeling workflow demonstrates the simplicity of the final API:

```julia
using BayesianFootball

1. Load Data from the Data module

matches = BayesianFootball.Data.load_matches("path/to/data.csv")
training_data, testing_data = BayesianFootball.Data.split_data(matches)

2. Define a model object from the Models module

my_model = BayesianFootball.Models.AR1NegativeBinomialHAModel()

3. Create features for that specific model type

training_features = BayesianFootball.Features.create_features(my_model, training_data)

4. Train the model using the generic Training API

chains = BayesianFootball.Training.train(my_model, training_features, training_data)

5. Make predictions on new data

testing_features = BayesianFootball.Features.create_features(my_model, testing_data)
predictions = BayesianFootball.Prediction.predict(my_model, chains, testing_features)

6. Evaluate the predictions

rps = BayesianFootball.Evaluation.calculate_rps(predictions, testing_data)
