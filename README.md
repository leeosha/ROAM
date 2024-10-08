# ROAM : ROI constrained Optimal Allocation Model


## Description

Sponsored search is crucial for e-commerce revenue, requiring a balance between maximizing platform revenue and maintaining user experience and advertiser utility. This paper introduces ROAM, a ROI-constrained allocation model that formulates the allocation problem as a constrained optimization task. It aims to maximize revenue while minimizing ad impressions, adhering to campaign budgets and ROI constraints. Utilizing a scalable iterative optimization algorithm within a parameter server framework, ROAM generates efficient allocation plans. Experiments on real-world data show significant improvements in both platform revenue and advertiser ROI.

## Table of Contents
- [Installation](#Installation)
- [Usage](#Usage)    
- [Feature](#Feature)

## Installation

* xdl

* tensorflow

## Usage

### input format：

* demand table：

    CREATE TABLE IF NOT EXISTS trip_ad_roam_demand
    
    (
    
        demand_hash_id   BIGINT COMMENT 'murmurhash(demandid)',
        
        demand_id        string,
        
        demand_name      string,
        
        alpha            DOUBLE comment 'init with 0',
        
        sigma            DOUBLE comment 'init with 0',
        
        theta            DOUBLE comment 'demand_volume/supply_volume',
        
        priority         DOUBLE,
        
        demand_pv        DOUBLE,
        
        uvsupply         DOUBLE,
        
        frequency        DOUBLE,
        
        batch_id         DOUBLE,
        
        price            DOUBLE,
        
        penality         DOUBLE,
        
        eta              double comment 'init with 0',
        
        zeta             double comment 'init with 0',
        
        target_roi_lower_bound       double,
        
        target_roi_upper_bound       double,
        
        roi_coeff        double
        
    );

* supply table：Order list corresponding to each request (in adjacency list format)

    create table if not exists trip_ad_roam_supply(
    
        sample_id        STRING COMMENT 'Sample ID: A hash value used for referencing in the clues, in a sparse structure.',
        
        sample_hash_id   STRING,
        
        supply_params    STRING COMMENT 'Traffic constraint parameter si, which is the node capacity: request_num.',
        

        ad_demand_ids    STRING COMMENT 'Ad ID, after hashing. Comma-separated.',
        
        ad_demand_freqs   STRING COMMENT 'Frequency control information. Comma-separated.',
        
        ad_demand_scores STRING COMMENT 'Revenue scores. Comma-separated.',
        
        ad_demand_ctrs   string comment 'Comma-separated.',
        
        ad_demand_cvrs   string comment 'Comma-separated.',
        
        ad_demand_cpcs   string comment 'Comma-separated.',
        
        ad_demand_prices string comment 'Comma-separated.'
        
    );


### output format：

* model table:

    CREATE TABLE IF NOT EXISTS trip_ad_roam_output
    
    (
        demand_hash_id   BIGINT,
        
        alpha            DOUBLE,
        
        sigma            DOUBLE,
        
        eta              double,
        
        zeta             double
        
    );

