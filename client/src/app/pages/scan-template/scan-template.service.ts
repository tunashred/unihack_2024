import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ScanTemplateService {
  private apiUrl = 'http://127.0.0.1:5000/process_image';

  constructor(private http: HttpClient) {}

  processImage(file: File, modelName: string): Observable<Blob> {
    const formData = new FormData();
    formData.append('image_file', file);
    formData.append('model_name', modelName);

    return this.http.post(this.apiUrl, formData, {
      responseType: 'blob' // Expect a binary response (image) instead of JSON
    });
  }
}
