//! Depth-limited regression decision tree for use as a weak learner in
//! gradient boosting.  Splits are chosen greedily to minimise MSE.

#[derive(Debug, Clone)]
pub enum Node {
    Leaf {
        value: f64,
    },
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl Node {
    pub fn predict(&self, features: &[f64]) -> f64 {
        match self {
            Node::Leaf { value } => *value,
            Node::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if features[*feature_index] <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }
}

/// Build a regression tree on (features, residuals) up to `max_depth`.
pub fn build_tree(
    features: &[Vec<f64>],
    targets: &[f64],
    max_depth: usize,
    min_samples_leaf: usize,
) -> Node {
    if max_depth == 0 || features.len() <= min_samples_leaf {
        return Node::Leaf {
            value: mean(targets),
        };
    }

    let n_features = features.first().map(|f| f.len()).unwrap_or(0);
    let mut best_loss = f64::INFINITY;
    let mut best_feat = 0;
    let mut best_thresh = 0.0f64;

    for fi in 0..n_features {
        // Collect unique thresholds for this feature.
        let mut vals: Vec<f64> = features.iter().map(|f| f[fi]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();

        for i in 0..vals.len().saturating_sub(1) {
            let thresh = (vals[i] + vals[i + 1]) / 2.0;
            let (l_idx, r_idx): (Vec<usize>, Vec<usize>) = (0..features.len())
                .partition(|&j| features[j][fi] <= thresh);

            if l_idx.len() < min_samples_leaf || r_idx.len() < min_samples_leaf {
                continue;
            }

            let l_targets: Vec<f64> = l_idx.iter().map(|&i| targets[i]).collect();
            let r_targets: Vec<f64> = r_idx.iter().map(|&i| targets[i]).collect();
            let loss = mse_total(&l_targets) + mse_total(&r_targets);

            if loss < best_loss {
                best_loss = loss;
                best_feat = fi;
                best_thresh = thresh;
            }
        }
    }

    if best_loss == f64::INFINITY {
        return Node::Leaf {
            value: mean(targets),
        };
    }

    let (l_idx, r_idx): (Vec<usize>, Vec<usize>) =
        (0..features.len()).partition(|&j| features[j][best_feat] <= best_thresh);

    let l_feat: Vec<Vec<f64>> = l_idx.iter().map(|&i| features[i].clone()).collect();
    let r_feat: Vec<Vec<f64>> = r_idx.iter().map(|&i| features[i].clone()).collect();
    let l_targ: Vec<f64> = l_idx.iter().map(|&i| targets[i]).collect();
    let r_targ: Vec<f64> = r_idx.iter().map(|&i| targets[i]).collect();

    Node::Split {
        feature_index: best_feat,
        threshold: best_thresh,
        left: Box::new(build_tree(&l_feat, &l_targ, max_depth - 1, min_samples_leaf)),
        right: Box::new(build_tree(&r_feat, &r_targ, max_depth - 1, min_samples_leaf)),
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn mse_total(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mu = mean(xs);
    xs.iter().map(|x| (x - mu).powi(2)).sum()
}
