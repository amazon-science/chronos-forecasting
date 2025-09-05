from chronosx.injection_blocks.input_injection_block import InputInjectionBlock
from chronosx.injection_blocks.output_injection_block import OutputInjectionBlock

injection_blocks_map = {
    "IIB": (InputInjectionBlock, None),
    "OIB": (None, OutputInjectionBlock),
    "IIB+OIB": (InputInjectionBlock, OutputInjectionBlock),
}
