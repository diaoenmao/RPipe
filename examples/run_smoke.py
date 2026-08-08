"""Example: run the smoke suite programmatically."""

from experiments import ResearchPipeline


if __name__ == '__main__':
    ResearchPipeline(suite_name='smoke', device='cpu').run()
