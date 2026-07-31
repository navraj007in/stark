# stark-env

`stark-env` provides read-only process argument and environment-variable access through the first-party native environment provider.

Implemented:

- bounded `args`;
- bounded `args_with_limits`;
- `get`;
- `get_with_limit`;
- `get_required`;
- explicit `EnvLimits`;
- absent and present-empty environment variables remain distinct;
- invalid environment names are rejected;
- invalid encoding is reported rather than replaced.

Excluded:

- environment mutation;
- process spawning;
- current-directory and executable-path APIs.
