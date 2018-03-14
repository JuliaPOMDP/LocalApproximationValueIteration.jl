"""
Base type for a local value function approximator
"""
abstract type LocalValueFnApproximator end


function n_interpolants end
function interpolating_states end
function interpolants end
function evaluate end
function batchUpdate end