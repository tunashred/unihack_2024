import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { ScanTemplateService } from '../../pages/scan-template/scan-template.service';

@Component({
  selector: 'file-uploader',
  standalone: true,
  imports: [],
  templateUrl: './file-uploader.component.html',
  styles: ``
})
export class FileUploaderComponent {
  constructor(private route: ActivatedRoute, private scanTemplateService: ScanTemplateService) {}
  public pageId: string | null = '';
  public images: any[] = [];
  public imageUrl: string | ArrayBuffer | null = null;  // Store the image URL

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.pageId = params.get('id');
    });
  }

  public saveFile(event: any) {
    const file = event.target.files[0];
    if (file) {
      this.scanTemplateService.processImage(file, 'kidney-cancer').subscribe(
        (res: Blob) => {
          // Create an Object URL for the image
          this.imageUrl = URL.createObjectURL(res);
        },
        (error) => {
          console.error('Error processing image:', error);
        }
      );
    }
  }
}
