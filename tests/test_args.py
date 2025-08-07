from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.finetune import FlatArguments
from open_instruct.utils import ArgumentParserPlus


class TestFlatArguments:
    def test_additional_model_args(self) -> None:
        parser = ArgumentParserPlus(FlatArguments)
        # NOTE: the boolean must be lower case, true not True
        (args,) = parser.parse_args_into_dataclasses(
            ["--additional_model_arguments", '{"int": 1, "bool": true, "float": 0.0, "float2": 5e-7}']
        )
        assert isinstance(args.additional_model_arguments, dict)
        assert isinstance(args.additional_model_arguments["int"], int)
        assert isinstance(args.additional_model_arguments["bool"], bool)
        assert isinstance(args.additional_model_arguments["float"], float)
        assert isinstance(args.additional_model_arguments["float2"], float)

    def test_no_additional_model_args(self) -> None:
        parser = ArgumentParserPlus(FlatArguments)
        # NOTE: the boolean must be lower case, true not True
        (args,) = parser.parse_args_into_dataclasses(["--exp_name", "test"])
        # Should get a empty dict
        assert isinstance(args.additional_model_arguments, dict)
        assert not args.additional_model_arguments


class TestTokenizerConfig:
    def test_transform_fn_kwargs(self) -> None:
        parser = ArgumentParserPlus(TokenizerConfig)
        # NOTE: the boolean must be lower case, true not True
        (args,) = parser.parse_args_into_dataclasses(
            ["--transform_fn_kwargs", '{"int": 1, "bool": true, "float": 0.0, "float2": 5e-7}']
        )
        assert isinstance(args.transform_fn_kwargs, dict)
        assert isinstance(args.transform_fn_kwargs["int"], int)
        assert isinstance(args.transform_fn_kwargs["bool"], bool)
        assert isinstance(args.transform_fn_kwargs["float"], float)
        assert isinstance(args.transform_fn_kwargs["float2"], float)

    def test_no_tranform_fn_kwargs(self) -> None:
        parser = ArgumentParserPlus(TokenizerConfig)
        # NOTE: the boolean must be lower case, true not True
        (args,) = parser.parse_args_into_dataclasses(["--tokenizer_name_or_path", "test"])
        # Should get a empty dict
        assert isinstance(args.transform_fn_kwargs, dict)
        assert not args.transform_fn_kwargs
