import unittest
from unittest.mock import patch, MagicMock

# Assuming 'scripts' is in PYTHONPATH or a way for Python to find it.
# If the script is run from the root, 'scripts.tipo' should be resolvable.
from scripts.tipo import TIPOScript, DEFAULT_FORMAT

# Path to kgen might also need to be adjusted or kgen installed as a package
# For these patches, we assume kgen.module.function_name is how it's imported/used

class TestTIPOScriptDeduplication(unittest.TestCase):

    @patch('modules.extra_networks.parse_prompt') # For extranet parsing at the start of _process
    @patch('kgen.formatter.seperate_tags')
    @patch('kgen.executor.tipo.parse_tipo_request')
    @patch('kgen.executor.tipo.tipo_runner')
    @patch('kgen.models.load_model')
    # apply_format is not called if no_formatting=True, so patching it is optional for this specific test path
    # but included for robustness if that flag changes.
    @patch('kgen.formatter.apply_format')
    def test_addon_tag_deduplication_single_run(self,
                                               mock_apply_format,
                                               mock_load_model,
                                               mock_tipo_runner,
                                               mock_parse_tipo_request,
                                               mock_seperate_tags,
                                               mock_parse_extranet_prompt): # Order matches @patch decorators from bottom up
        script = TIPOScript()

        user_prompt_tags = "user_tag1, user_tag2"
        nl_user_prompt = "Test NL prompt"

        # Mock for extranet parsing (parse_prompt from modules.extra_networks)
        # This is the first call in _process using the main prompt.
        mock_parse_extranet_prompt.return_value = (user_prompt_tags, {}) # (prompt_without_extranet, res)

        # Mock for seperate_tags used to create org_tag_map and all_normalized_user_tags
        # This is called with all_tags derived from prompt_parse_strength of user_prompt_tags.
        # For simplicity, assume prompt_parse_strength yields user_prompt_tags directly.
        mock_org_tag_map_from_user = {
            "general": ["user_tag1", "user_tag2"],
            "quality": [], "meta": [], "rating": [], "special": [],
            "copyrights": [], "artist": [], "characters": [], "extended": [], "generated": []
        }
        mock_seperate_tags.return_value = mock_org_tag_map_from_user

        # Mock for parse_tipo_request
        # Input to parse_tipo_request: org_tag_map, nl_prompt (initial_nl_text_for_kgen), ...
        # initial_nl_text_for_kgen is derived from the nl_prompt argument to _process, after its own parsing.
        # Let's assume nl_prompt_processed_text (becomes initial_nl_text_for_kgen) is same as nl_user_prompt.
        mock_parse_tipo_request.return_value = (
            {"aspect_ratio": "1.0"}, # meta
            [], # operations (empty list for default behavior)
            mock_org_tag_map_from_user.get("general", []), # general_tags_for_runner
            nl_user_prompt # nl_prompt_for_runner
        )

        # Configure the mock tipo_runner to return a tag_map with duplicates
        mock_model_output_tag_map = {
            "general": ["new_tag1", "user_tag1", "new_tag2", "new_tag1", "New_Tag2_Caps", "numeric123", "UPPER_TAG"],
            "character": ["char_tag1", "char_tag1", "123numeric", "upper_tag"], # "upper_tag" will be a dupe of "UPPER_TAG" from general after normalization
            "extended": "This is some extended text.",
            "quality": ["quality_tag"], "meta": ["meta_tag"], "rating": ["rating_tag"],
            "special": ["special_tag"], "copyrights": ["copyright_tag"], "artist": ["artist_tag"],
            "generated": "", "characters":[]
        }
        mock_tipo_runner.return_value = (mock_model_output_tag_map, {"some_meta_key": "some_value"})

        # Call _process with relevant arguments
        result_prompt = script._process(
            prompt=user_prompt_tags,
            nl_prompt=nl_user_prompt,
            aspect_ratio=1.0, seed=1234,
            tag_length="long", # Ensures single run mode for this test
            min_tags=10, max_tags=50,
            nl_length="long", ban_tags="banned_tag",
            format_select="AnyFormat", format_str=DEFAULT_FORMAT,
            temperature=0.5, top_p=0.9, top_k=50,
            ignore_first_n_tags=0, model="mock_model_name",
            gguf_use_cpu=False,
            no_formatting=True, # Key: no_formatting=True to get unformatted output
            tag_prompt=user_prompt_tags # Fallback if prompt is empty
        )

        expected_addon_tags_content_sorted = sorted([
            "new_tag1", "new_tag2", "UPPER_TAG",
            "char_tag1", "123numeric",
            "quality_tag", "meta_tag", "rating_tag",
            "special_tag", "copyright_tag", "artist_tag"
        ])
        expected_addon_nl = "This is some extended text."

        cleaned_result_prompt = result_prompt.strip()

        # The _process method joins with "\n", so we split on that exact sequence.
        parts = cleaned_result_prompt.split('\\n')
        main_tag_line = parts[0]
        result_addon_nl = parts[1] if len(parts) > 1 else ""

        self.assertEqual(result_addon_nl, expected_addon_nl, "NL part mismatch")

        prefix_to_remove = user_prompt_tags + ", "
        if main_tag_line.startswith(prefix_to_remove):
            actual_addon_tags_str = main_tag_line[len(prefix_to_remove):]
        elif main_tag_line == user_prompt_tags:
            actual_addon_tags_str = ""
        else:
            self.fail(f"Main tag line '{main_tag_line}' does not start with user prompt tags '{prefix_to_remove}' as expected.")

        actual_addon_tags_list_sorted = sorted([tag.strip() for tag in actual_addon_tags_str.split(',') if tag.strip()])

        self.assertListEqual(actual_addon_tags_list_sorted, expected_addon_tags_content_sorted, "Addon tags list mismatch")

if __name__ == '__main__':
    unittest.main()
