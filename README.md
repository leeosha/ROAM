# ROAM: ROI-constrained Optimal Allocation Model

## Description

Sponsored search is a critical component of e-commerce revenue, requiring a delicate balance between maximizing platform revenue and maintaining user experience and advertiser utility. This repository introduces ROAM, a ROI-constrained allocation model that formulates the allocation problem as a constrained optimization task. ROAM aims to maximize revenue while minimizing ad impressions, adhering to campaign budgets and ROI constraints. By utilizing a scalable iterative optimization algorithm within a parameter server framework, ROAM generates efficient allocation plans. Experiments conducted on real-world data demonstrate significant improvements in both platform revenue and advertiser ROI.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Note](#note)

## Installation

To use ROAM, you'll need to install the following dependencies:

* XDL (X-DeepLearning)
  
  For installation instructions, visit: https://github.com/alibaba/x-deeplearning

* TensorFlow

  For installation instructions, visit: https://github.com/tensorflow/tensorflow

## Usage

### Running the Code

1. Install the required dependencies as per the installation instructions.
2. Prepare your dataset according to the format specified in the [Dataset](#dataset) section.
3. Execute `main.py` to run the model.

### Input Format

ROAM requires two input tables:

1. Demand Table (`trip_ad_roam_demand`):
   Contains information about advertising demands, including demand IDs, priorities, volumes, and ROI constraints.

2. Supply Table (`trip_ad_roam_supply`):
   Represents the order list corresponding to each request in an adjacency list format, including sample IDs, traffic constraints, and ad-related information.

For detailed schema information, please refer to the SQL CREATE TABLE statements in the original README.

### Output Format

The model outputs its results to the `trip_ad_roam_output` table, which includes the following columns:

- `demand_hash_id`: Hashed demand ID
- `alpha`: Optimized alpha parameter
- `sigma`: Optimized sigma parameter
- `eta`: Optimized eta parameter
- `zeta`: Optimized zeta parameter

## Dataset

### Instructions

The dataset used for this project is shared via Baidu Netdisk.

- Link: https://pan.baidu.com/s/1othj1qrTFBR6nCNd-Ogjlw?pwd=tx6f 
- Access Code: tx6f

### Format

The dataset is structured as follows:

- Column 0: user_id (int)
- Column 1: query_id (int)
- Column 2: Sample source (list, 0: advertising, 1: organic)
- Column 3: Sample label (list)
- Columns 4+: Side information [itemID, categoryID, brandID, vendorID, priceID, display_count, click_count, budget, and min_ROI] for each sample (list)
- Column 5: user_category_id (int)
- Column 6: timestamp (int)

## Note

This repository accompanies the research paper titled "ROI Constrained Optimal Online Allocation in Sponsored Search". It provides the implementation of the ROAM model and the necessary resources to reproduce the results presented in the paper.

For any questions or issues, please open an issue in this repository or contact the authors directly.
