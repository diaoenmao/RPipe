"""Keep script exports used by existing generated launchers."""
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.location, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_make,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('name', [
    'repo_root_from', 'bash_path', 'job_waves', 'scripts_dir', 'write_jobs_json',
    'wait_groups', 'render_bash', 'write_launch_scripts',
])
def test_existing_schedule_exports_remain_direct(name):
    from rpipe.structure.make import schedule, scripts
    assert getattr(schedule, name) is getattr(scripts, name)
