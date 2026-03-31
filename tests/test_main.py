import sys
import pytest
from main import parse, match_args, main

@pytest.mark.parametrize("environment, ebool", [
        (["-e", "connectfour"], True),
        (["-e", "connectfou"], False),
        (["--env", "connectfour"], True),
        (["--env", "connectfou"], False),
        (None, True)
    ])
@pytest.mark.parametrize("player, pbool", [
        (["-p", "qAgent"], True),
        (["-p", "randAgent"], True),
        (["-p", "randAgen"], False),
        (["--player", "qAgent"], True),
        (["--player", "randAgent"], True),
        (["--player", "randAgen"], False),
        (None, True)
    ])
class TestParser:

    def test_match_args_exception(self, environment, ebool, player, pbool):
        args, noExcept = self.convert_args(environment, ebool, player, pbool)
        try:
            match_args(parse(args))
        except Exception as e:
            assert noExcept is False
        else:
            assert noExcept is True

    def convert_args(self, environment, ebool, player, pbool):
        args = []
        if environment:
            args += environment
        if player:
            args += player
        
        if args is not None:
            args = (a for a in args if a is not None)
        return args, (ebool and pbool)

        