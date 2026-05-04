"""Dangerous pattern detection for tool inputs.

This implements detection of potentially dangerous operations
that warrant additional scrutiny or user confirmation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


# =============================================================================
# Danger Categories
# =============================================================================


class DangerCategory(str, Enum):
    """Categories of dangerous operations."""

    IRREVERSIBLE_DESTRUCTION = "irreversible_destruction"
    CODE_FROM_EXTERNAL = "code_from_external"
    UNAUTHORIZED_PERSISTENCE = "unauthorized_persistence"
    SECURITY_WEAKEN = "security_weaken"
    EXFILTRATION_RISK = "exfiltration_risk"
    PRIVACY_LEAK = "privacy_leak"
    FINANCIAL_IMPACT = "financial_impact"


@dataclass
class DangerDetection:
    """Result of danger detection."""

    is_dangerous: bool
    category: Optional[DangerCategory] = None
    severity: str = "medium"  # low, medium, high, critical
    reason: Optional[str] = None
    pattern_matched: Optional[str] = None


# =============================================================================
# Bash Dangerous Patterns
# =============================================================================


# Irreversible destruction patterns
DESTRUCTION_PATTERNS = [
    # rm -rf variants
    (r"rm\s+-rf\s+", "rm -rf (recursive forced delete)"),
    (r"rm\s+-fr\s+", "rm -fr (recursive forced delete)"),
    (r"rm\s+--recursive\s+--force", "rm --recursive --force"),
    (r"rm\s+-r\s+--force", "rm -r --force"),
    (r"rmdir\s+/s\s+/q", "rmdir /s /q (Windows recursive delete)"),
    (r"del\s+/s\s+/q", "del /s /q (Windows recursive delete)"),
    (r"Remove-Item\s+-Recurse\s+-Force", "Remove-Item -Recurse -Force (PowerShell)"),

    # Disk/system wiping
    (r"mkfs\.", "mkfs (format filesystem)"),
    (r"dd\s+if=.*of=/dev/", "dd to device (disk overwrite)"),
    (r">/dev/sd[a-z]", "overwrite disk device"),
    (r"wipefs", "wipe filesystem signature"),
    (r"shred", "shred (secure delete)"),

    # Truncation/overwrite
    (r">\s+[^/]", "truncate file (> redirection)"),
    (r">>\s*truncate", "truncate command"),

    # Kill processes (potentially dangerous)
    (r"kill\s+-9\s+1", "kill -9 init (system crash)"),
    (r"killall\s+", "killall (kill all matching processes)"),
    (r"pkill\s+-9", "pkill -9 (force kill matching)"),

    # Database operations
    (r"DROP\s+(TABLE|DATABASE|SCHEMA)", "DROP TABLE/DATABASE/SCHEMA"),
    (r"TRUNCATE\s+TABLE", "TRUNCATE TABLE"),
    (r"DELETE\s+FROM\s+", "DELETE FROM (bulk delete)"),
]

# Code from external patterns
EXTERNAL_CODE_PATTERNS = [
    # Curl/wget pipe to shell
    (r"curl\s+[^|]*\|\s*(bash|sh|zsh)", "curl | shell (download and execute)"),
    (r"wget\s+[^|]*\|\s*(bash|sh|zsh)", "wget | shell (download and execute)"),
    (r"fetch\s+[^|]*\|\s*(bash|sh|zsh)", "fetch | shell (download and execute)"),

    # pip/npm from URL
    (r"pip\s+install\s+--index-url\s+", "pip install from custom URL"),
    (r"pip\s+install\s+git+", "pip install from git URL"),
    (r"npm\s+install\s+https?:", "npm install from URL"),
    (r"npm\s+install\s+git+", "npm install from git URL"),

    # Download and execute variants
    (r"curl\s+.*>\s+.*&&\s+chmod\s+", "download + chmod + execute"),
    (r"wget\s+.*>\s+.*&&\s+chmod\s+", "download + chmod + execute"),
    (r"Invoke-WebRequest\s+.*\|\s*iex", "PowerShell download and execute"),
    (r"Invoke-Expression\s+.*\(Invoke-WebRequest", "PowerShell IEX from web"),
    (r"iwr\s+.*\|\s*iex", "PowerShell IWR | IEX"),

    # Git clone and execute
    (r"git\s+clone\s+.*&&\s+.*(?:run|execute|bash)", "git clone and execute"),

    # Python eval/exec from URL
    (r"eval\s*\(\s*(?:urllib|requests|httpx)", "Python eval from web request"),
    (r"exec\s*\(\s*(?:urllib|requests|httpx)", "Python exec from web request"),
]

# Unauthorized persistence patterns
PERSISTENCE_PATTERNS = [
    # Shell config modification
    (r">\s*~/.bashrc", "overwrite .bashrc"),
    (r">\s*~/.zshrc", "overwrite .zshrc"),
    (r">\s*~/.profile", "overwrite .profile"),
    (r">\s*~/.bash_profile", "overwrite .bash_profile"),
    (r"echo\s+.*>\s*~/.bashrc", "append to .bashrc"),
    (r"echo\s+.*>\s*~/.zshrc", "append to .zshrc"),

    # Cron jobs
    (r"crontab\s+-e", "edit crontab"),
    (r">\s*/etc/cron", "write to cron directory"),
    (r"echo\s+.*>\s*/etc/cron", "add cron job"),

    # System services
    (r"systemctl\s+enable", "enable systemd service"),
    (r"chkconfig\s+--add", "add init.d service"),
    (r"launchctl\s+load", "load launchd service"),
    (r"sc\s+create", "create Windows service"),
    (r"New-Service", "PowerShell create service"),

    # Scheduled tasks
    (r"schtasks\s+/create", "create scheduled task"),
    (r"Register-ScheduledTask", "PowerShell scheduled task"),

    # Registry run keys
    (r"reg\s+add\s+.*\\\\Run", "add registry Run key"),
    (r"Set-ItemProperty\s+.*HK.*Run", "PowerShell registry Run key"),

    # WMI event subscriptions
    (r"Register-WmiEvent", "PowerShell WMI event"),

    # PowerShell profile
    (r">\s*\$PROFILE", "overwrite PowerShell profile"),
    (r"echo\s+.*>\s*\$PROFILE", "append to PowerShell profile"),
]

# Security weakening patterns
SECURITY_WEAKEN_PATTERNS = [
    # Disable firewall
    (r"ufw\s+disable", "disable UFW firewall"),
    (r"iptables\s+-F", "flush iptables rules"),
    (r"iptables\s+-X", "delete iptables chains"),
    (r"firewall-cmd\s+--disable", "disable firewalld"),
    (r"netsh\s+advfirewall\s+set\s+.*off", "disable Windows firewall"),
    (r"Set-NetFirewallProfile\s+-Enabled\s+False", "PowerShell disable firewall"),

    # Disable antivirus/security
    (r"Set-MpPreference\s+-DisableRealtimeMonitoring", "disable Defender"),
    (r"chkconfig\s+.*off", "disable service (chkconfig)"),

    # Lower permissions
    (r"chmod\s+777", "chmod 777 (everyone full access)"),
    (r"chmod\s+-R\s+777", "chmod -R 777 (recursive)"),
    (r"chmod\s+000", "chmod 000 (no permissions)"),
    (r"chown\s+.*:.*\s+/", "change ownership of root path"),
    (r"setfacl\s+-m\s+u:nobody:rw", "ACL modification"),

    # Disable SELinux/AppArmor
    (r"setenforce\s+0", "disable SELinux enforcement"),
    (r"aa-disable", "disable AppArmor profile"),

    # SSH/security config
    (r">\s*/etc/ssh/sshd_config", "overwrite sshd config"),
    (r"PermitRootLogin\s+yes", "enable root SSH login"),
    (r"PasswordAuthentication\s+yes", "enable SSH password auth"),

    # Clear logs
    (r">\s*/var/log/", "truncate log file"),
    (r"rm\s+/var/log/", "remove log file"),
    (r"auditctl\s+-D", "delete audit rules"),
]

# Exfiltration risk patterns
EXFILTRATION_PATTERNS = [
    # Upload to external
    (r"curl\s+.*-T\s+.*(?:https?|ftp)", "curl -T upload to external"),
    (r"scp\s+.*(?:@|https?|ftp)", "scp to remote server"),
    (r"rsync\s+.*(?:@|https?|ftp)", "rsync to remote"),
    (r"sftp\s+.*(?:put|upload)", "sftp upload"),
    (r"wget\s+.*--post-file", "wget POST file upload"),

    # Send secrets
    (r"(?:cat|head|tail)\s+.*(?:\.env|\.pem|\.key|\.secret|credentials)\s*\|\s*(?:curl|wget|nc)",
     "send credential file externally"),
    (r"nc\s+.*<\s+.*(?:\.env|\.pem|\.key)", "netcat send credential file"),

    # API key exposure
    (r"echo\s+\$.*(?:API_KEY|SECRET|PASSWORD|TOKEN)\s*\|\s*(?:curl|wget)",
     "send environment secret externally"),
]

# Privacy leak patterns
PRIVACY_LEAK_PATTERNS = [
    # Expose sensitive directories
    (r"ls\s+-la\s+~", "list home directory contents"),
    (r"find\s+~/.*-type\s+f", "find files in home directory"),
    (r"cat\s+~/.(?:ssh|gnupg|config)", "read private config"),

    # Read password files
    (r"cat\s+/etc/shadow", "read shadow file (password hashes)"),
    (r"cat\s+/etc/passwd", "read passwd file"),
    (r"getent\s+shadow", "get shadow entries"),

    # Read browser data
    (r"cat\s+.*(?:cookies|history|passwords).*(?:db|sqlite|json)",
     "read browser data"),
    (r"sqlite3\s+.*(?:cookies|history)", "query browser database"),

    # Read wallet/crypto
    (r"cat\s+.*(?:wallet|keystore|\.eth)", "read crypto wallet"),
]

# Financial impact patterns
FINANCIAL_PATTERNS = [
    # Cloud cost operations
    (r"aws\s+.*(?:create|run|launch|start)", "AWS create/start resource"),
    (r"gcloud\s+.*(?:create|run|deploy)", "GCP create/deploy"),
    (r"az\s+.*(?:create|vm\s+create)", "Azure create resource"),

    # Terraform/Ansible apply
    (r"terraform\s+apply", "terraform apply (create resources)"),
    (r"ansible\s+.*(?:deploy|apply)", "ansible deploy"),

    # Docker compose up (creates containers)
    (r"docker\s+compose\s+up\s+-d", "docker compose up (creates resources)"),
    (r"docker\s+run\s+", "docker run (creates container)"),

    # Kubernetes create
    (r"kubectl\s+create", "kubectl create resource"),
    (r"kubectl\s+apply", "kubectl apply (creates/updates)"),
    (r"helm\s+install", "helm install chart"),
]


# =============================================================================
# Detection Functions
# =============================================================================


def check_command_dangerous(command: str) -> DangerDetection:
    """Check if a bash command is dangerous.

    Args:
        command: Bash command string

    Returns:
        Danger detection result
    """
    # Normalize command
    normalized = command.strip()

    # Check persistence FIRST (before destruction truncation patterns)
    # This ensures .bashrc, .zshrc modifications are caught as persistence
    for pattern, description in PERSISTENCE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return DangerDetection(
                is_dangerous=True,
                category=DangerCategory.UNAUTHORIZED_PERSISTENCE,
                severity="high",
                reason=f"Detected persistence modification: {description}",
                pattern_matched=pattern,
            )

    # Check destruction patterns
    for pattern, description in DESTRUCTION_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return DangerDetection(
                is_dangerous=True,
                category=DangerCategory.IRREVERSIBLE_DESTRUCTION,
                severity="critical",
                reason=f"Detected destructive operation: {description}",
                pattern_matched=pattern,
            )

    # Check external code patterns
    for pattern, description in EXTERNAL_CODE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return DangerDetection(
                is_dangerous=True,
                category=DangerCategory.CODE_FROM_EXTERNAL,
                severity="critical",
                reason=f"Detected code from external source: {description}",
                pattern_matched=pattern,
            )

    for pattern, description in SECURITY_WEAKEN_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return DangerDetection(
                is_dangerous=True,
                category=DangerCategory.SECURITY_WEAKEN,
                severity="high",
                reason=f"Detected security weakening: {description}",
                pattern_matched=pattern,
            )

    for pattern, description in EXFILTRATION_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return DangerDetection(
                is_dangerous=True,
                category=DangerCategory.EXFILTRATION_RISK,
                severity="high",
                reason=f"Detected potential data exfiltration: {description}",
                pattern_matched=pattern,
            )

    for pattern, description in PRIVACY_LEAK_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return DangerDetection(
                is_dangerous=True,
                category=DangerCategory.PRIVACY_LEAK,
                severity="medium",
                reason=f"Detected privacy-sensitive access: {description}",
                pattern_matched=pattern,
            )

    for pattern, description in FINANCIAL_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return DangerDetection(
                is_dangerous=True,
                category=DangerCategory.FINANCIAL_IMPACT,
                severity="medium",
                reason=f"Detected financial impact operation: {description}",
                pattern_matched=pattern,
            )

    # No danger detected
    return DangerDetection(is_dangerous=False)


def check_file_write_dangerous(file_path: str, content: str = "") -> DangerDetection:
    """Check if a file write operation is dangerous.

    Args:
        file_path: Target file path
        content: Content to write (optional)

    Returns:
        Danger detection result
    """
    # Check for system configuration files
    dangerous_paths = [
        "/etc/ssh/sshd_config",
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "/etc/cron",
        "/etc/systemd",
        "~/.ssh/authorized_keys",
        "~/.ssh/id_rsa",
        "~/.bashrc",
        "~/.zshrc",
        "~/.profile",
    ]

    for path in dangerous_paths:
        if file_path.endswith(path) or file_path == path:
            return DangerDetection(
                is_dangerous=True,
                category=DangerCategory.SECURITY_WEAKEN,
                severity="high",
                reason=f"Writing to sensitive file: {path}",
            )

    # Check content for dangerous patterns
    if content:
        content_danger = check_command_dangerous(content)
        if content_danger.is_dangerous:
            return DangerDetection(
                is_dangerous=True,
                category=content_danger.category,
                severity=content_danger.severity,
                reason=f"Writing dangerous content: {content_danger.reason}",
            )

    return DangerDetection(is_dangerous=False)


def check_tool_input_dangerous(tool_name: str, input: Any) -> DangerDetection:
    """Check if a tool input is dangerous.

    Args:
        tool_name: Tool name
        input: Tool input (validated)

    Returns:
        Danger detection result
    """
    # Bash tool - check command
    if tool_name == "Bash":
        command = getattr(input, "command", "")
        if isinstance(command, str):
            return check_command_dangerous(command)

    # Write tool - check file path
    if tool_name == "Write":
        file_path = getattr(input, "file_path", "")
        content = getattr(input, "content", "")
        return check_file_write_dangerous(file_path, content)

    # Edit tool - check file path
    if tool_name == "Edit":
        file_path = getattr(input, "file_path", "")
        return check_file_write_dangerous(file_path)

    # Default - not dangerous
    return DangerDetection(is_dangerous=False)


# =============================================================================
# Severity Helpers
# =============================================================================


def get_severity_description(severity: str) -> str:
    """Get human-readable severity description.

    Args:
        severity: Severity level

    Returns:
        Description
    """
    descriptions = {
        "critical": "CRITICAL - Irreversible damage possible",
        "high": "HIGH - Significant risk, requires explicit approval",
        "medium": "MEDIUM - Moderate risk, should confirm",
        "low": "LOW - Minor risk, proceed with caution",
    }
    return descriptions.get(severity, severity)


def is_critical_danger(detection: DangerDetection) -> bool:
    """Check if detection is critical severity.

    Args:
        detection: Danger detection

    Returns:
        True if critical
    """
    return detection.severity == "critical"


def should_block_without_confirmation(detection: DangerDetection) -> bool:
    """Check if operation should be blocked without asking.

    Args:
        detection: Danger detection

    Returns:
        True if should block
    """
    return detection.severity in ("critical", "high")


def requires_explicit_user_confirmation(detection: DangerDetection) -> bool:
    """Check if operation requires explicit user confirmation.

    Args:
        detection: Danger detection

    Returns:
        True if requires confirmation
    """
    return detection.severity in ("critical", "high", "medium")