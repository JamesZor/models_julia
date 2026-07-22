
using Revise
using BayesianFootball
using DataFrames
using Distributions
using ThreadPinning
using ProgressMeter

pinthreads(:cores)


const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Data        = BayesianFootball.Data
const Signals     = BayesianFootball.Signals


ds = Data.load_datastore_cached(Data.Ireland())
src_dir   = "./data/double_poisson_smile_src_grid/"   # NEW: this runner's src cells



# load_experiments
list_of_experiments = Experiments.list_experiments(src_dir, data_dir="")
expr = Experiments.load_experiment(list_of_experiments, 5)

split_id = 25

split = BayesianFootball.Data.create_data_splits(ds, expr.config.splitter)[split_id]


expr_split = expr.training_results[split_id]

next_matches = BayesianFootball.Data.get_next_matches(ds, expr_split, expr.config.splitter)

all_expr_inference = BayesianFootball.Predictions.model_inference(ds, expr) 

