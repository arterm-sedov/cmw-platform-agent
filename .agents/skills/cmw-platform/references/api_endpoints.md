# API Endpoints

## Base URL
```
{CMW_BASE_URL}/webapi/{Endpoint}
```

CMW_BASE_URL from configuration (platform-specific).

## Authentication

HTTP Basic Auth:
```
Authorization: Basic {base64(login:password)}
```

Credentials from configuration: `CMW_LOGIN`, `CMW_PASSWORD`

## Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| webapi/Solution | GET | List applications |
| webapi/RecordTemplate/List/{app} | GET | List record templates |
| webapi/ProcessTemplate/List/{app} | GET | List process templates |
| webapi/AccountTemplate/List/{app} | GET | List account templates |
| webapi/Attribute/List/Template@{app}.{tmpl} | GET | List template attributes |
| webapi/Records/Template@{app}.{tmpl} | GET | List records (with pagination) |
| webapi/Records/Template@{app}.{tmpl} | POST | Create record |
| webapi/Record/{template_global_alias} | POST | Create record via Record endpoint |
| webapi/Record/{record_id} | PUT | Update existing record |
| webapi/Record/{record_id} | DELETE | Delete record |

## Request/Response Format

### Request (POST/PUT)
```json
Content-Type: application/json
{
  "AttributeSystemName": "value"
}
```

### Response (Success)
```json
{
  "response": {...} | [...],
  "success": true,
  "error": null
}
```

### Response (Error)
```json
{
  "response": null,
  "success": false,
  "error": {"message": "...", "type": "..."}
}
```

## HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid params) |
| 401 | Unauthorized (bad credentials) |
| 408 | Request timeout |
| 500 | Internal server error |

## Template Global Alias

Format: `Template@{application_system_name}.{template_system_name}`

Example: `Template@MyApp.MyTemplate`