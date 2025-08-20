# Core morphism functions
function features_morphism(model_config::ModelConfig, mapping::MappedData)
    return data -> model_config.feature_map(data, mapping)
end

# Morphism: Features -> TuringModels
function models_morphism(model_config::ModelConfig)
    return features -> begin
        ht_model = model_config.model(
            features.home_team_ids,
            features.away_team_ids,
            features.goals_home_ht,
            features.goals_away_ht,
            features.n_teams
        )
        
        ft_model = model_config.model(
            features.home_team_ids,
            features.away_team_ids,
            features.goals_home_ft,
            features.goals_away_ft,
            features.n_teams
        )
        
        BasicMaherModels(ht_model, ft_model)
    end
end

# TODO: add number of chains to config
# Morphism: TuringModels -> Chains 
function sampling_morphism(sample_config::ModelSampleConfig)
    return models -> begin
        ht_chain = sample(models.ht, NUTS(), MCMCSerial(), 
                         sample_config.steps, 1; 
                         progress=sample_config.bar)
        ft_chain = sample(models.ft, NUTS(), MCMCSerial(), 
                         sample_config.steps, 1;
                         progress=sample_config.bar)
        ModelChain(ht_chain, ft_chain)
    end
end
# Composed training morphism: SubDataFrame -> TrainedChains
function compose_training_morphism(
    model_config::ModelConfig,
    sample_config::ModelSampleConfig,
    mapping::MappedData
)
    # Compose the morphisms: data -> features -> models -> chains
    return (data, info) -> begin
        features = features_morphism(model_config, mapping)(data)
        models = models_morphism(model_config)(features)
        chains = sampling_morphism(sample_config)(models)
        
        TrainedChains(
            chains.ht,
            chains.ft,
            info,
            nrow(data)
        )
    end
end
