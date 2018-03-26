"""
Base type for a local value function approximator
"""
abstract type LocalValueFunctionApproximator end


function n_interpolants end
function get_all_interpolating_states end
function get_all_interpolants end
function get_interpolating_nbrs_idxs_wts end
function evaluate end
function batchUpdate end