module ReactantNNlibExt

using NNlib
using Reactant

for (jlop, hloop) in ((:(NNlib.tanh), :tanh), (:(NNlib.tanh_fast), :tanh))
    @eval begin
        if $jlop != Base.tanh && $jlop != Base.FastMath.tanh_fast
            function Reactant.elem_apply(
                ::typeof($jlop), lhs::Reactant.TracedRArray{ElType,Shape,N}
            ) where {ElType,Shape,N}
                return Reactant.TracedRArray{ElType,Shape,N}(
                    (),
                    Reactant.MLIR.IR.result(
                        Reactant.MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1
                    ),
                )
            end
        end
    end
end

# TODO handle non finite cases
function NNlib.softmax!(
    out::Reactant.TracedRArray{T,Shape,N}, x::AbstractArray; dims=1
) where {T,Shape,N}
    max_ = NNlib.fast_maximum(x; dims)
    #if all(isfinite, max_)
    @fastmath out .= exp.(x .- max_)
    #else
    #    _zero, _one, _inf = T(0), T(1), T(Inf)
    #    @fastmath @. out = ifelse(isequal(max_,_inf), ifelse(isequal(x,_inf), _one, _zero), exp(x - max_))
    #end
    tmp = dims isa Colon ? sum(out) : sum!(max_, out)
    return out ./= tmp
end

function NNlib.conv(
    in1::Reactant.TracedRArray{T,Shape1,4}, in2::Reactant.TracedRArray{T,Shape2,4}, cdims::C
) where {T,Shape1,Shape2,C<:NNlib.DenseConvDims}
    W, H, _, B = Shape1
    kw, kh = NNlib.kernel_size(cdims)
    pw, ph, _, _ = NNlib.padding(cdims)
    dw, dh = NNlib.dilation(cdims)
    sw, sh = NNlib.stride(cdims)

    h_out = (H + 2 * ph - dh * (kh - 1) - 1) ÷ sh + 1
    w_out = (W + 2 * pw - dw * (kw - 1) - 1) ÷ sw + 1
    c_out = Shape2[4]
    output_size = (w_out, h_out, c_out, B)
    output_type = Reactant.MLIR.IR.TensorType(output_size, Reactant.MLIR.IR.Type(T))

    window_strides = [sw, sh]
    rhs_dilation = [dw, dh]
    lhs_dilation = [1, 1]
    padding = [
        pw pw
        ph ph
    ]

    in1, in2 = in1.mlir_data, in2.mlir_data
    if !NNlib.flipkernel(cdims)
        rev = Reactant.MLIR.Dialects.stablehlo.reverse(in2; dimensions=[1,0])
        in2 = Reactant.MLIR.IR.result(rev, 1)
    end

    return Reactant.TracedRArray{T,output_size,4}(
        (),
        Reactant.MLIR.IR.result(
            Reactant.MLIR.Dialects.stablehlo.convolution(
                in1, in2;
                result_0=output_type,
                dimension_numbers=parse(
                    Reactant.MLIR.IR.Attribute,
                    """
                    #stablehlo.conv<raw
                        input_batch_dimension = 3,
                        input_feature_dimension = 2,
                        input_spatial_dimensions = [0, 1],
                        kernel_input_feature_dimension = 2,
                        kernel_output_feature_dimension = 3,
                        kernel_spatial_dimensions = [0, 1],
                        output_batch_dimension = 3,
                        output_feature_dimension = 2,
                        output_spatial_dimensions = [0, 1]
                    >
                    """,
                ),
                rhs_dilation,
                lhs_dilation,
                padding, window_strides,
                feature_group_count=1,
                batch_group_count=1,
            ),
            1,
        ),
    )
end

function NNlib.maxpool(
    in0::Reactant.TracedRArray{T,Shape,N}, dims::NNlib.PoolDims
) where {T,Shape,N}
    window_dimensions = [NNlib.kernel_size(dims)..., 1, 1]
    window_strides = [NNlib.stride(dims)..., 1, 1]
    base_dilations = [1, 1, 1, 1]
    window_dilations = [NNlib.dilation(dims)..., 1, 1]
    padding = [NNlib.padding(dims)..., 0, 0][begin:4]
    padding = repeat(padding; inner=(1, 2))

    W, H, C, B = Shape

    kw, kh = NNlib.kernel_size(dims)
    pw, ph, _, _ = NNlib.padding(dims)
    dw, dh = NNlib.dilation(dims)
    sw, sh = NNlib.stride(dims)
    h_out = (H + 2 * ph - dh * (kh - 1) - 1) ÷ sh + 1
    w_out = (W + 2 * pw - dw * (kw - 1) - 1) ÷ sw + 1

    output_size = (w_out, h_out, C, B)
    output_type = Reactant.MLIR.IR.TensorType(output_size, Reactant.MLIR.IR.Type(T))

    unranked_tensor_type = Reactant.MLIR.IR.TensorType((), Reactant.MLIR.IR.Type(T))
    inner_block = let
        inner_block = Reactant.MLIR.IR.Block(
            [unranked_tensor_type, unranked_tensor_type],
            [Reactant.MLIR.IR.Location(), Reactant.MLIR.IR.Location()],
        )
        Reactant.MLIR.IR.block!(inner_block) do
            val = Reactant.MLIR.IR.argument(inner_block, 1)
            init = Reactant.MLIR.IR.argument(inner_block, 2)
            maxop = Reactant.MLIR.Dialects.stablehlo.maximum(val, init)
            Reactant.MLIR.Dialects.stablehlo.return_([Reactant.MLIR.IR.result(maxop, 1)])
        end
        inner_block
    end
    body = Reactant.MLIR.IR.Region()
    push!(body, inner_block)

    init_value = Reactant.MLIR.IR.result(
        Reactant.MLIR.Dialects.stablehlo.constant(;
            value=Base.fill(typemin(T), unranked_tensor_type)
        ),
        1,
    )

    reduce_window = Reactant.MLIR.Dialects.stablehlo.reduce_window(
        [in0.mlir_data],
        [init_value];
        result_0=[output_type],
        window_dimensions,
        window_strides,
        base_dilations,
        window_dilations,
        padding,
        body,
    )

    return Reactant.TracedRArray{T,output_size,N}(
        (), Reactant.MLIR.IR.result(reduce_window, 1)
    )
end

end # module ReactantNNlibExt
