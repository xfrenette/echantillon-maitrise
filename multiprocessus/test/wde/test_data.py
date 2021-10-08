import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, ContextManager, Dict, List, Tuple

from random_words import LoremIpsum

from evaluate.with_domain_embedding import DomainsDataSource


@contextmanager
def _create_tmp_files(domains: Sequence[str], source_extension=".en", target_extension=".fr", other_extensions=(".meta",),
                      nb_sentences=15) -> ContextManager[Tuple[Path, Dict[str, Dict[str, List[str]]]]]:
    sentences = {}
    lipsum = LoremIpsum()
    with TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)
        for domain in domains:
            sentences[domain] = {}
            src_path = tmp_dir / (domain + source_extension)
            tgt_path = tmp_dir / (domain + target_extension)

            with open(src_path, "w") as f:
                file_sentences = lipsum.get_sentences_list(nb_sentences)
                sentences[domain]["source"] = file_sentences
                f.write("\n".join(file_sentences) + "\n")

            with open(tgt_path, "w") as f:
                file_sentences = lipsum.get_sentences_list(nb_sentences)
                sentences[domain]["target"] = file_sentences
                f.write("\n".join(file_sentences) + "\n")

            for other_ext in other_extensions:
                path = tmp_dir / (domain + other_ext)
                with open(path, "w") as f:
                    f.write("\n".join(lipsum.get_sentences_list(nb_sentences)) + "\n")

        yield tmp_dir, sentences


def _rewrite_files(tmp_dir: Path, nb_sentences=10):
    """
    Completely change the content of all files in a directory
    :param tmp_dir: The directory
    """
    lipsum = LoremIpsum()
    for file_path in tmp_dir.glob("*"):
        file_sentences = lipsum.get_sentences_list(nb_sentences)
        with open(file_path, "w") as f:
            f.write("\n".join(file_sentences) + "\n")


class DomainsDataSourceTestCase(unittest.TestCase):
    def test_returns_expected_sentences(self):
        domains = ["dom1", "dom2"]
        src_ext = ".a"
        tgt_ext = ".b"
        with _create_tmp_files(domains, src_ext, tgt_ext) as (tmp_dir, sentences):
            data_source = DomainsDataSource(tmp_dir, src_ext, tgt_ext)

            for domain in domains:
                src_sent = data_source.get_source_sentences(domain)
                self.assertListEqual(sentences[domain]["source"], src_sent)

                tgt_sent = data_source.get_target_sentences(domain)
                self.assertListEqual(sentences[domain]["target"], tgt_sent)

    def test_use_cache(self):
        for use_cache in [True, False]:
            with self.subTest(use_cache=use_cache):
                with _create_tmp_files(["dom"]) as (tmp_dir, _):
                    data_source = DomainsDataSource(tmp_dir)

                    with data_source.use_cache(use_cache):
                        # Get the original value in the files
                        source_sentences_orig = data_source.get_source_sentences("dom")
                        target_sentences_orig = data_source.get_target_sentences("dom")

                        # Change the content of the files
                        _rewrite_files(tmp_dir)

                        # Call again to get sentences
                        source_sentences_new = data_source.get_source_sentences("dom")
                        target_sentences_new = data_source.get_target_sentences("dom")

                        # If we use cache, the new sentences should be the same as the orig sentences. If we don't use
                        # the cache, the new sentences should have changed
                        if use_cache:
                            self.assertListEqual(source_sentences_orig, source_sentences_new)
                            self.assertListEqual(target_sentences_orig, target_sentences_new)
                        else:
                            self.assertNotEqual(source_sentences_orig, source_sentences_new)
                            self.assertNotEqual(target_sentences_orig, target_sentences_new)

    def test_recall_use_cache(self):
        """Cache should be cleared between calls of use_cache"""
        with _create_tmp_files(["dom"]) as (tmp_dir, _):
            data_source = DomainsDataSource(tmp_dir)

            with data_source.use_cache():
                # Get the original value in the files
                source_sentences_orig = data_source.get_source_sentences("dom")
                target_sentences_orig = data_source.get_target_sentences("dom")

            # Change the content of the files
            _rewrite_files(tmp_dir)

            with data_source.use_cache():
                # Call again to get sentences
                source_sentences_new = data_source.get_source_sentences("dom")
                target_sentences_new = data_source.get_target_sentences("dom")

            # Cache should be cleared between calls of `use_cache()`, so the sentences should have changed
            self.assertNotEqual(source_sentences_orig, source_sentences_new)
            self.assertNotEqual(target_sentences_orig, target_sentences_new)

    def test_max_cache_size(self):
        """
        use_cache() should accept a maximum cache size

        If we set the cache size to 2 and we call `get_*_sentences` for 3 different sets, the first one should be
        cleared. If we change the sentences after the first 3 calls, calling again for the first set should now return
        new content, but the other 2 should still return the same content.
        """
        with _create_tmp_files(["dom1", "dom2"]) as (tmp_dir, _):
            data_source = DomainsDataSource(tmp_dir)

            with data_source.use_cache(max_cache_size=2):
                set1_orig = data_source.get_source_sentences("dom1")
                set2_orig = data_source.get_source_sentences("dom2")
                set3_orig = data_source.get_target_sentences("dom1")

                _rewrite_files(tmp_dir)

                # The order here is important ! The set1_new must be last since it was the first to be called (else, it
                # would overwrite the cache of set2, and set2 would overwrite the cache of set3!)
                set2_new = data_source.get_source_sentences("dom2")
                set3_new = data_source.get_target_sentences("dom1")
                set1_new = data_source.get_source_sentences("dom1")

                self.assertListEqual(set2_orig, set2_new)
                self.assertListEqual(set3_orig, set3_new)
                self.assertNotEqual(set1_orig, set1_new)

    def test_raises_for_invalid_domain(self):
        with _create_tmp_files(("dom1",)) as (tmp_dir, _):
            data_source = DomainsDataSource(tmp_dir)

            with self.assertRaises(FileNotFoundError):
                data_source.get_source_sentences("invalid-domain")

            with self.assertRaises(FileNotFoundError):
                data_source.get_target_sentences("invalid-domain")

    def test_domains(self):
        """Test the `domains` property"""
        domains = ("dom1", "dom2", "dom3", "dom4", "dom5")
        with _create_tmp_files(domains) as (tmp_dir, _):
            # We delete the target file of dom3 to make it an invalid domain
            file_to_delete: Path = tmp_dir / "dom3.fr"
            file_to_delete.unlink()

            # We delete the source file of dom4 to make it an invalid domain
            file_to_delete: Path = tmp_dir / "dom4.en"
            file_to_delete.unlink()

            data_source = DomainsDataSource(tmp_dir)
            expected = {"dom1", "dom2", "dom5"}
            actual = set(data_source.domains)
            self.assertSetEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
